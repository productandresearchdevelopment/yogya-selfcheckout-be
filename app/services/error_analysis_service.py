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
from app.services.evaluation_service import compute_iou
from app.schemas.evaluation import (
    ErrorAnalysisResponse,
    PerSkuMetric,
    ConfusionMatrixData,
    ErrorExampleRecord,
)


def run_error_analysis(
    db: Session,
    model_id: Optional[str] = None,
    split: str = "val",
    similarity_threshold: float = 0.60,
    margin_threshold: float = 0.05,
) -> ErrorAnalysisResponse:
    """
    Executes a structured Error Analysis pipeline categorizing recognition failures:
    - CORRECT_KNOWN
    - CORRECT_UNKNOWN
    - WRONG_SKU_MISCLASSIFICATION
    - FALSE_POSITIVE_UNKNOWN_AS_KNOWN
    - FALSE_REJECTION_KNOWN_AS_UNKNOWN
    - DETECTOR_MISSED_OBJECT
    - DETECTOR_POOR_CROP
    - EMBEDDING_INFERENCE_FAILURE
    - VERSION_MISMATCH
    - LOW_QUALITY_OCCLUDED

    Separates detector failures from visual embedding matching per Section 4 Requirement 3.
    Generates confusion matrix, per-SKU precision/recall/F1, and representative error cases.
    """
    if model_id:
        model_rec = db.query(ModelRegistry).filter(ModelRegistry.model_id == model_id).first()
        if not model_rec:
            raise ValueError(f"Model ID '{model_id}' not found in registry.")
    else:
        model_rec = get_active_model(db, model_type="embedding")

    model, device, transform = load_model_instance(model_rec)

    # 1. Build Reference Gallery for this model
    active_skus = db.query(SKU).filter(SKU.is_active == True).order_by(SKU.sku_code).all()
    sku_id_to_obj = {s.id: s for s in active_skus}
    sku_code_to_obj = {s.sku_code: s for s in active_skus}

    ref_images = (
        db.query(ProductImage)
        .join(SKU)
        .filter(ProductImage.role == "reference", SKU.is_active == True)
        .all()
    )

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
                    "dim": len(vec),
                })
        except Exception:
            pass

    # 2. Fetch Query Evaluation Images for target split
    # For split evaluation, we query images explicitly matching the split or role
    query_images = (
        db.query(ProductImage)
        .filter(
            ProductImage.split == split,
            ProductImage.role.in_(["validation", "test", "training", "query"]),
            ProductImage.role != "reference",  # Never evaluate reference gallery against itself
        )
        .all()
    )

    if not query_images:
        # Fallback if split column wasn't explicitly assigned
        role_target = "validation" if split == "val" else "test"
        query_images = (
            db.query(ProductImage)
            .filter(ProductImage.role == role_target)
            .all()
        )

    # 3. Categorize Queries and Compute Predictions
    category_counts: Dict[str, int] = {
        "CORRECT_KNOWN": 0,
        "CORRECT_UNKNOWN": 0,
        "WRONG_SKU_MISCLASSIFICATION": 0,
        "FALSE_POSITIVE_UNKNOWN_AS_KNOWN": 0,
        "FALSE_REJECTION_KNOWN_AS_UNKNOWN": 0,
        "DETECTOR_MISSED_OBJECT": 0,
        "DETECTOR_POOR_CROP": 0,
        "EMBEDDING_INFERENCE_FAILURE": 0,
        "VERSION_MISMATCH": 0,
        "LOW_QUALITY_OCCLUDED": 0,
    }

    # Per-SKU tracking
    sku_stats: Dict[str, Dict[str, Any]] = {}
    for s in active_skus:
        sku_stats[s.sku_code] = {
            "sku_id": s.id,
            "sku_name": s.name,
            "total_queries": 0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "similarities": [],
        }

    # Confusion matrix labels
    labels = [s.sku_code for s in active_skus] + ["UNKNOWN"]
    label_to_idx = {l: idx for idx, l in enumerate(labels)}
    confusion_matrix = [[0 for _ in range(len(labels))] for _ in range(len(labels))]

    representative_errors: List[ErrorExampleRecord] = []
    total_known = 0
    total_unknown = 0

    detector = get_detector()

    for q_img in query_images:
        if not os.path.exists(q_img.file_path):
            continue

        gt_sku_code = q_img.sku.sku_code if q_img.sku else "UNKNOWN"
        is_ground_truth_unknown = (q_img.sku_id is None or gt_sku_code == "UNKNOWN")

        if is_ground_truth_unknown:
            total_unknown += 1
        else:
            total_known += 1
            if gt_sku_code in sku_stats:
                sku_stats[gt_sku_code]["total_queries"] += 1

        gt_idx = label_to_idx.get(gt_sku_code, label_to_idx["UNKNOWN"])

        # Check Step A: Image Quality / Occlusion
        if (q_img.blur_score is not None and q_img.blur_score < 30.0) or q_img.width < 32 or q_img.height < 32:
            category_counts["LOW_QUALITY_OCCLUDED"] += 1
            representative_errors.append(ErrorExampleRecord(
                query_id=q_img.id,
                filename=q_img.filename,
                image_url=f"/api/v1/datasets/images/{q_img.id}/file",
                ground_truth_label=gt_sku_code,
                predicted_label="UNKNOWN",
                similarity_score=0.0,
                margin_delta=0.0,
                error_category="LOW_QUALITY_OCCLUDED",
                reason=f"Severe blur/low resolution: blur_score={q_img.blur_score or 0.0:.1f}, {q_img.width}x{q_img.height}",
            ))
            pred_idx = label_to_idx["UNKNOWN"]
            confusion_matrix[gt_idx][pred_idx] += 1
            continue

        # Check Step B: Detector Crop Quality (if annotated ground truth box exists)
        det_error = False
        det_conf_found = None
        if q_img.annotations:
            gt_box = (
                q_img.annotations[0].x_min,
                q_img.annotations[0].y_min,
                q_img.annotations[0].x_max,
                q_img.annotations[0].y_max,
            )
            try:
                with Image.open(q_img.file_path) as pil_img:
                    dets = detector.detect_objects(pil_img, confidence_threshold=0.30)
                if not dets or all(d.get("is_fallback", False) for d in dets):
                    category_counts["DETECTOR_MISSED_OBJECT"] += 1
                    det_error = True
                    representative_errors.append(ErrorExampleRecord(
                        query_id=q_img.id,
                        filename=q_img.filename,
                        image_url=f"/api/v1/datasets/images/{q_img.id}/file",
                        ground_truth_label=gt_sku_code,
                        predicted_label="UNKNOWN",
                        similarity_score=0.0,
                        margin_delta=0.0,
                        error_category="DETECTOR_MISSED_OBJECT",
                        reason="Object detector produced 0 bounding boxes on annotated object.",
                    ))
                else:
                    best_iou = max(compute_iou(d["box"], gt_box) for d in dets if not d.get("is_fallback", False))
                    det_conf_found = max(d["confidence"] for d in dets if not d.get("is_fallback", False))
                    if best_iou < 0.40:
                        category_counts["DETECTOR_POOR_CROP"] += 1
                        det_error = True
                        representative_errors.append(ErrorExampleRecord(
                            query_id=q_img.id,
                            filename=q_img.filename,
                            image_url=f"/api/v1/datasets/images/{q_img.id}/file",
                            ground_truth_label=gt_sku_code,
                            predicted_label="UNKNOWN",
                            similarity_score=0.0,
                            margin_delta=0.0,
                            error_category="DETECTOR_POOR_CROP",
                            detector_confidence=det_conf_found,
                            reason=f"Detector bounding box IoU={best_iou:.2f} < 0.40 against ground-truth.",
                        ))
            except Exception:
                pass

        if det_error:
            pred_idx = label_to_idx["UNKNOWN"]
            confusion_matrix[gt_idx][pred_idx] += 1
            continue

        # Check Step C: Embedding Extraction
        try:
            with Image.open(q_img.file_path) as img:
                img = img.convert("RGB")
                tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_vec = model(tensor).squeeze().cpu().numpy().astype(np.float32)
                norm = np.linalg.norm(q_vec)
                if norm <= 1e-6 or np.isnan(norm):
                    category_counts["EMBEDDING_INFERENCE_FAILURE"] += 1
                    representative_errors.append(ErrorExampleRecord(
                        query_id=q_img.id,
                        filename=q_img.filename,
                        image_url=f"/api/v1/datasets/images/{q_img.id}/file",
                        ground_truth_label=gt_sku_code,
                        predicted_label="UNKNOWN",
                        similarity_score=0.0,
                        margin_delta=0.0,
                        error_category="EMBEDDING_INFERENCE_FAILURE",
                        reason="Zero-norm or NaN vector output during embedding inference.",
                    ))
                    pred_idx = label_to_idx["UNKNOWN"]
                    confusion_matrix[gt_idx][pred_idx] += 1
                    continue
                q_vec = q_vec / norm
        except Exception as e:
            category_counts["EMBEDDING_INFERENCE_FAILURE"] += 1
            pred_idx = label_to_idx["UNKNOWN"]
            confusion_matrix[gt_idx][pred_idx] += 1
            continue

        # Check Step D: Model/Gallery Version Dimension Mismatch
        if gallery and len(q_vec) != gallery[0]["dim"]:
            category_counts["VERSION_MISMATCH"] += 1
            representative_errors.append(ErrorExampleRecord(
                query_id=q_img.id,
                filename=q_img.filename,
                image_url=f"/api/v1/datasets/images/{q_img.id}/file",
                ground_truth_label=gt_sku_code,
                predicted_label="UNKNOWN",
                similarity_score=0.0,
                margin_delta=0.0,
                error_category="VERSION_MISMATCH",
                reason=f"Embedding dimension {len(q_vec)} differs from gallery dimension {gallery[0]['dim']}.",
            ))
            pred_idx = label_to_idx["UNKNOWN"]
            confusion_matrix[gt_idx][pred_idx] += 1
            continue

        # Step E: Cosine Similarity Matching & Margin Threshold Decision Rule
        similarities: List[Tuple[float, str]] = []
        for g_item in gallery:
            sim = float(np.dot(q_vec, g_item["vector"]))
            similarities.append((sim, g_item["sku_code"]))

        similarities.sort(key=lambda x: x[0], reverse=True)

        top1_sim, top1_sku_code = similarities[0] if similarities else (0.0, "UNKNOWN")
        top2_sim = similarities[1][0] if len(similarities) > 1 else 0.0
        margin_delta = round(top1_sim - top2_sim, 4)

        # Decision Rule: Accepted if sim >= similarity_threshold AND margin_delta >= margin_threshold
        is_accepted = (top1_sim >= similarity_threshold) and (margin_delta >= margin_threshold)
        predicted_label = top1_sku_code if is_accepted else "UNKNOWN"

        pred_idx = label_to_idx.get(predicted_label, label_to_idx["UNKNOWN"])
        confusion_matrix[gt_idx][pred_idx] += 1

        # Track per-SKU similarity
        if not is_ground_truth_unknown and gt_sku_code in sku_stats:
            sku_stats[gt_sku_code]["similarities"].append(top1_sim)

        # Categorize Outcome
        if is_ground_truth_unknown:
            if is_accepted:
                # UNKNOWN object falsely accepted as a known SKU
                category_counts["FALSE_POSITIVE_UNKNOWN_AS_KNOWN"] += 1
                if top1_sku_code in sku_stats:
                    sku_stats[top1_sku_code]["false_positives"] += 1
                representative_errors.append(ErrorExampleRecord(
                    query_id=q_img.id,
                    filename=q_img.filename,
                    image_url=f"/api/v1/datasets/images/{q_img.id}/file",
                    ground_truth_label="UNKNOWN",
                    predicted_label=top1_sku_code,
                    similarity_score=round(top1_sim, 4),
                    margin_delta=margin_delta,
                    error_category="FALSE_POSITIVE_UNKNOWN_AS_KNOWN",
                    reason=f"Unknown object accepted as '{top1_sku_code}' with sim={top1_sim:.4f} >= {similarity_threshold:.2f}",
                ))
            else:
                # UNKNOWN correctly rejected
                category_counts["CORRECT_UNKNOWN"] += 1
        else:
            if is_accepted:
                if top1_sku_code == gt_sku_code:
                    category_counts["CORRECT_KNOWN"] += 1
                    sku_stats[gt_sku_code]["true_positives"] += 1
                else:
                    category_counts["WRONG_SKU_MISCLASSIFICATION"] += 1
                    sku_stats[gt_sku_code]["false_negatives"] += 1
                    if top1_sku_code in sku_stats:
                        sku_stats[top1_sku_code]["false_positives"] += 1
                    representative_errors.append(ErrorExampleRecord(
                        query_id=q_img.id,
                        filename=q_img.filename,
                        image_url=f"/api/v1/datasets/images/{q_img.id}/file",
                        ground_truth_label=gt_sku_code,
                        predicted_label=top1_sku_code,
                        similarity_score=round(top1_sim, 4),
                        margin_delta=margin_delta,
                        error_category="WRONG_SKU_MISCLASSIFICATION",
                        reason=f"True SKU '{gt_sku_code}' misclassified as '{top1_sku_code}' (sim={top1_sim:.4f})",
                    ))
            else:
                # Known product falsely rejected as UNKNOWN
                category_counts["FALSE_REJECTION_KNOWN_AS_UNKNOWN"] += 1
                sku_stats[gt_sku_code]["false_negatives"] += 1
                reason_str = (
                    f"Sim={top1_sim:.4f} < {similarity_threshold:.2f}"
                    if top1_sim < similarity_threshold
                    else f"Margin {margin_delta:.4f} < {margin_threshold:.2f}"
                )
                representative_errors.append(ErrorExampleRecord(
                    query_id=q_img.id,
                    filename=q_img.filename,
                    image_url=f"/api/v1/datasets/images/{q_img.id}/file",
                    ground_truth_label=gt_sku_code,
                    predicted_label="UNKNOWN",
                    similarity_score=round(top1_sim, 4),
                    margin_delta=margin_delta,
                    error_category="FALSE_REJECTION_KNOWN_AS_UNKNOWN",
                    reason=f"Known SKU '{gt_sku_code}' falsely rejected as UNKNOWN ({reason_str})",
                ))

    # 4. Compute Per-SKU Metrics
    per_sku_results: List[PerSkuMetric] = []
    for code, stats in sku_stats.items():
        tp = stats["true_positives"]
        fp = stats["false_positives"]
        fn = stats["false_negatives"]
        tot = stats["total_queries"]

        prec = round(tp / max(1, tp + fp), 4) if (tp + fp) > 0 else 0.0
        rec = round(tp / max(1, tp + fn), 4) if (tp + fn) > 0 else 0.0
        f1 = round(2 * (prec * rec) / max(1e-6, prec + rec), 4) if (prec + rec) > 0 else 0.0
        acc = round(tp / max(1, tot), 4) if tot > 0 else 0.0
        mean_sim = round(float(np.mean(stats["similarities"])), 4) if stats["similarities"] else 0.0

        per_sku_results.append(PerSkuMetric(
            sku_id=stats["sku_id"],
            sku_code=code,
            sku_name=stats["sku_name"],
            total_queries=tot,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            precision=prec,
            recall=rec,
            f1_score=f1,
            accuracy=acc,
            mean_similarity=mean_sim,
        ))

    # 5. Summarize Overall Metrics
    total_samples = len(query_images)
    overall_correct = category_counts["CORRECT_KNOWN"] + category_counts["CORRECT_UNKNOWN"]
    top1_accuracy = round(category_counts["CORRECT_KNOWN"] / max(1, total_known), 4) if total_known > 0 else 0.0
    unknown_far = round(category_counts["FALSE_POSITIVE_UNKNOWN_AS_KNOWN"] / max(1, total_unknown), 4) if total_unknown > 0 else 0.0
    known_frr = round(category_counts["FALSE_REJECTION_KNOWN_AS_UNKNOWN"] / max(1, total_known), 4) if total_known > 0 else 0.0

    return ErrorAnalysisResponse(
        status="COMPLETED",
        model_id=model_rec.model_id,
        split_evaluated=split,
        similarity_threshold=similarity_threshold,
        margin_threshold=margin_threshold,
        total_samples=total_samples,
        total_known_samples=total_known,
        total_unknown_samples=total_unknown,
        category_breakdown=category_counts,
        per_sku_metrics=per_sku_results,
        confusion_matrix=ConfusionMatrixData(labels=labels, matrix=confusion_matrix),
        representative_errors=representative_errors[:50],
        detector_metrics_separated={
            "detector_model": "FasterRCNN-MobileNetV3 FPN",
            "detector_missed_count": category_counts["DETECTOR_MISSED_OBJECT"],
            "detector_poor_crop_count": category_counts["DETECTOR_POOR_CROP"],
            "note": "Detector errors isolated from embedding similarity calculations per Requirement 3.",
        },
        summary={
            "top1_accuracy": top1_accuracy,
            "overall_accuracy": round(overall_correct / max(1, total_samples), 4) if total_samples > 0 else 0.0,
            "unknown_false_acceptance_rate": unknown_far,
            "known_false_rejection_rate": known_frr,
            "error_rate": round(1.0 - (overall_correct / max(1, total_samples)), 4) if total_samples > 0 else 0.0,
        },
    )
