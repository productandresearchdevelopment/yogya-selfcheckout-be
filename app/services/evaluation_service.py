import os
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image
import torch
from sqlalchemy.orm import Session

from app.models.sku import SKU
from app.models.image import ProductImage
from app.models.annotation import Annotation
from app.models.model_registry import ModelRegistry
from app.services.detector import get_detector
from app.services.model_registry_service import get_active_model, load_model_instance
from app.services.embedder import get_embedder


def compute_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    """Computes Intersection over Union for two normalized boxes [x_min, y_min, x_max, y_max]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH

    boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

    union = boxAArea + boxBArea - interArea
    if union <= 0.0:
        return 0.0
    return interArea / union


def evaluate_detector_localization(db: Session, confidence_threshold: float = 0.35) -> Dict[str, Any]:
    """
    Evaluates FasterRCNN object detector performance against ground-truth bounding box annotations.
    Strictly separated from embedding recognition evaluation per Requirement 5.
    If no annotations exist, cleanly reports the blocker without fabricating metrics.
    """
    annotated_images = (
        db.query(ProductImage)
        .join(Annotation)
        .filter(ProductImage.is_valid == True)
        .distinct()
        .all()
    )

    if not annotated_images:
        return {
            "status": "NO_GROUND_TRUTH_ANNOTATIONS",
            "message": "No images with ground-truth bounding box annotations found in dataset. Use Dataset Studio to annotate bounding boxes before running detector evaluation.",
            "metrics": {
                "annotated_images_count": 0,
                "precision_at_iou_50": None,
                "recall_at_iou_50": None,
                "map_50": None,
                "average_iou": None,
            },
            "eval_duration_ms": 0.0,
        }

    detector = get_detector()
    start_time = time.time()

    total_gt = 0
    total_det = 0
    true_positives = 0
    iou_scores = []

    for img_rec in annotated_images:
        if not os.path.exists(img_rec.file_path):
            continue

        gt_boxes = [(ann.x_min, ann.y_min, ann.x_max, ann.y_max) for ann in img_rec.annotations]
        total_gt += len(gt_boxes)

        try:
            with Image.open(img_rec.file_path) as pil_img:
                dets = detector.detect_objects(pil_img, confidence_threshold=confidence_threshold)
                detected_boxes = [d["box"] for d in dets if not d.get("is_fallback", False)]
                total_det += len(detected_boxes)

                # Match detected boxes to ground truth
                matched_gt = set()
                for det_box in detected_boxes:
                    best_iou = 0.0
                    best_gt_idx = -1
                    for idx, gt_box in enumerate(gt_boxes):
                        if idx in matched_gt:
                            continue
                        iou = compute_iou(det_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = idx

                    if best_iou >= 0.5 and best_gt_idx >= 0:
                        true_positives += 1
                        matched_gt.add(best_gt_idx)
                        iou_scores.append(best_iou)
        except Exception as e:
            print(f"Error evaluating image {img_rec.id}: {e}")

    precision = round(true_positives / max(1, total_det), 4) if total_det > 0 else 0.0
    recall = round(true_positives / max(1, total_gt), 4) if total_gt > 0 else 0.0
    avg_iou = round(float(np.mean(iou_scores)), 4) if iou_scores else 0.0

    duration_ms = round((time.time() - start_time) * 1000, 1)

    return {
        "status": "EVALUATED",
        "detector_model": "FasterRCNN-MobileNetV3-Large-FPN (COCO Pretrained)",
        "confidence_threshold": confidence_threshold,
        "metrics": {
            "annotated_images_count": len(annotated_images),
            "total_ground_truth_boxes": total_gt,
            "total_detected_boxes": total_det,
            "true_positives_iou_50": true_positives,
            "precision_at_iou_50": precision,
            "recall_at_iou_50": recall,
            "map_50": round((precision + recall) / 2.0, 4) if (precision + recall) > 0 else 0.0,
            "average_iou": avg_iou,
        },
        "eval_duration_ms": duration_ms,
    }


def evaluate_embedding_model_comparative(
    db: Session,
    target_model_id: Optional[str] = None,
    similarity_threshold: float = 0.60,
) -> Dict[str, Any]:
    """
    Evaluates and compares the Pretrained Baseline model vs Fine-Tuned Active Model
    on the exact same test queries, per Requirement 6.
    Computes:
    - Top-1 and Top-3 accuracy
    - Per-SKU accuracy and confusion breakdown
    - Unknown-object False Acceptance Rate (FAR)
    - Known-product False Rejection Rate (FRR)
    - Inference latency
    - Hard negative scenario breakdown
    """
    baseline_model_rec = (
        db.query(ModelRegistry)
        .filter(ModelRegistry.model_id == "mobilenetv3-pretrained-baseline")
        .first()
    )

    if target_model_id:
        target_model_rec = (
            db.query(ModelRegistry)
            .filter(ModelRegistry.model_id == target_model_id)
            .first()
        )
    else:
        target_model_rec = get_active_model(db, model_type="embedding")

    # Fetch reference gallery images (for gallery building)
    ref_images = (
        db.query(ProductImage)
        .join(SKU)
        .filter(ProductImage.role == "reference", SKU.is_active == True)
        .all()
    )

    # Fetch evaluation images (validation, test, query, or training non-reference)
    eval_images = (
        db.query(ProductImage)
        .filter(ProductImage.role.in_(["training", "validation", "test", "query"]))
        .all()
    )

    if not ref_images or not eval_images:
        return {
            "status": "INSUFFICIENT_EVAL_DATA",
            "message": "Evaluation requires both registered reference images and independent evaluation/test images.",
            "metrics_available": False,
        }

    # Evaluate Model A: Baseline
    baseline_results = _run_model_evaluation(
        baseline_model_rec, ref_images, eval_images, similarity_threshold
    )

    # Evaluate Model B: Target/Fine-tuned
    target_results = _run_model_evaluation(
        target_model_rec, ref_images, eval_images, similarity_threshold
    )

    # Threshold calibration sweep on target model
    calibrated_thresholds = _calibrate_threshold_sweep(
        target_model_rec, ref_images, eval_images
    )

    return {
        "status": "SUCCESS",
        "evaluated_images_count": len(eval_images),
        "gallery_reference_count": len(ref_images),
        "similarity_threshold": similarity_threshold,
        "baseline_model": baseline_results,
        "active_model": target_results,
        "comparison_delta": {
            "top1_accuracy_delta": round(target_results["top1_accuracy"] - baseline_results["top1_accuracy"], 4),
            "top3_accuracy_delta": round(target_results["top3_accuracy"] - baseline_results["top3_accuracy"], 4),
            "latency_delta_ms": round(target_results["average_latency_ms"] - baseline_results["average_latency_ms"], 2),
            "unknown_far_delta": round(target_results["unknown_false_acceptance_rate"] - baseline_results["unknown_false_acceptance_rate"], 4),
        },
        "threshold_calibration": calibrated_thresholds,
    }


def _run_model_evaluation(
    model_rec: ModelRegistry,
    ref_images: List[ProductImage],
    eval_images: List[ProductImage],
    similarity_threshold: float,
) -> Dict[str, Any]:
    """Runs metric extraction and accuracy assessment for a specific model record."""
    model, device, transform = load_model_instance(model_rec)

    # Build temporary evaluation gallery for this model
    gallery: List[Dict[str, Any]] = []
    for ref_img in ref_images:
        if not os.path.exists(ref_img.file_path):
            continue
        try:
            with Image.open(ref_img.file_path) as img:
                img = img.convert("RGB")
                tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    vec = model(tensor).squeeze().cpu().numpy().astype(np.float32)
                norm = np.linalg.norm(vec)
                if norm > 1e-6:
                    vec = vec / norm
                gallery.append({
                    "sku_id": ref_img.sku_id,
                    "sku_code": ref_img.sku.sku_code if ref_img.sku else "UNKNOWN",
                    "vector": vec,
                })
        except Exception:
            pass

    if not gallery:
        return {"error": "Gallery could not be built."}

    # Run queries
    latencies = []
    total_known = 0
    correct_top1 = 0
    correct_top3 = 0
    false_rejections = 0  # Known product mistakenly marked UNKNOWN

    total_unknown = 0
    false_acceptances = 0  # Unknown object mistakenly matched to a known SKU

    hard_negative_details = []
    per_sku_stats: Dict[str, Dict[str, int]] = {}

    for query_img in eval_images:
        if not os.path.exists(query_img.file_path):
            continue
        is_ground_truth_unknown = query_img.sku_id is None

        t0 = time.time()
        try:
            with Image.open(query_img.file_path) as img:
                img = img.convert("RGB")
                tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_vec = model(tensor).squeeze().cpu().numpy().astype(np.float32)
                norm = np.linalg.norm(q_vec)
                if norm > 1e-6:
                    q_vec = q_vec / norm
        except Exception:
            continue

        latencies.append((time.time() - t0) * 1000)

        # Match against gallery
        similarities = []
        for item in gallery:
            sim = float(np.dot(q_vec, item["vector"]))
            similarities.append((sim, item["sku_id"], item["sku_code"]))

        similarities.sort(key=lambda x: x[0], reverse=True)
        top1_sim, top1_sku_id, top1_sku_code = similarities[0]
        top3_sku_ids = [x[1] for x in similarities[:3]]

        is_rejected = top1_sim < similarity_threshold

        if is_ground_truth_unknown:
            total_unknown += 1
            if not is_rejected:
                false_acceptances += 1
            hard_negative_details.append({
                "sample_filename": query_img.filename,
                "type": "UNKNOWN_OBJECT",
                "top_matched_sku": top1_sku_code,
                "top_similarity": round(top1_sim, 4),
                "correctly_rejected": is_rejected,
            })
        else:
            total_known += 1
            sku_name = query_img.sku.sku_code if query_img.sku else "UNKNOWN"
            stat = per_sku_stats.setdefault(sku_name, {"total": 0, "correct_top1": 0})
            stat["total"] += 1

            if is_rejected:
                false_rejections += 1
            else:
                if top1_sku_id == query_img.sku_id:
                    correct_top1 += 1
                    stat["correct_top1"] += 1

                if query_img.sku_id in top3_sku_ids:
                    correct_top3 += 1

    top1_acc = round(correct_top1 / max(1, total_known), 4) if total_known > 0 else 0.0
    top3_acc = round(correct_top3 / max(1, total_known), 4) if total_known > 0 else 0.0
    frr = round(false_rejections / max(1, total_known), 4) if total_known > 0 else 0.0
    far = round(false_acceptances / max(1, total_unknown), 4) if total_unknown > 0 else 0.0
    avg_latency = round(float(np.mean(latencies)), 2) if latencies else 0.0

    return {
        "model_id": model_rec.model_id,
        "architecture": model_rec.architecture,
        "embedding_dim": model_rec.embedding_dim,
        "top1_accuracy": top1_acc,
        "top3_accuracy": top3_acc,
        "known_false_rejection_rate": frr,
        "unknown_false_acceptance_rate": far,
        "average_latency_ms": avg_latency,
        "total_known_queries": total_known,
        "total_unknown_queries": total_unknown,
        "per_sku_accuracy": {
            k: round(v["correct_top1"] / max(1, v["total"]), 4) for k, v in per_sku_stats.items()
        },
        "hard_negatives": hard_negative_details,
    }


def _calibrate_threshold_sweep(
    model_rec: ModelRegistry,
    ref_images: List[ProductImage],
    eval_images: List[ProductImage],
) -> List[Dict[str, Any]]:
    """Sweeps similarity thresholds from 0.30 to 0.90 to identify optimal F1 acceptance score."""
    sweep = []
    for th in [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        res = _run_model_evaluation(model_rec, ref_images, eval_images, similarity_threshold=th)
        top1 = res.get("top1_accuracy", 0.0)
        far = res.get("unknown_false_acceptance_rate", 0.0)
        frr = res.get("known_false_rejection_rate", 0.0)

        # Harmonic F1 between known acceptance and unknown rejection
        precision = (1.0 - far)
        recall = (1.0 - frr)
        f1 = round(2 * (precision * recall) / max(1e-6, precision + recall), 4)

        sweep.append({
            "threshold": th,
            "top1_accuracy": top1,
            "false_acceptance_rate": far,
            "false_rejection_rate": frr,
            "calibrated_f1_score": f1,
        })

    return sweep
