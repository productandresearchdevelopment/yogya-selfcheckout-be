import os
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image
import torch
from sqlalchemy.orm import Session

from app.models.sku import SKU
from app.models.image import ProductImage
from app.models.model_registry import ModelRegistry
from app.services.model_registry_service import get_active_model, load_model_instance
from app.schemas.evaluation import (
    ThresholdCalibrationRequest,
    ThresholdCalibrationResponse,
    ThresholdSweepPoint,
)


def run_threshold_calibration(
    db: Session,
    request: ThresholdCalibrationRequest,
) -> ThresholdCalibrationResponse:
    """
    Executes a repeatable threshold calibration sweep strictly using the validation split.
    Prohibits calibrating against the test set to prevent dataset leakage per Section 4 Requirement 5.

    Sweeps similarity thresholds (e.g. 0.35..0.85) and top-1 vs top-2 margin thresholds (e.g. 0.0..0.15).
    Selects optimal thresholds based on configurable objective:
    - min_far_with_recall_target: Minimize unknown FAR subject to known recall >= target
    - max_f1: Maximize harmonic mean between known precision and recall
    - balanced_eer: Minimize difference between FAR and FRR
    """
    if request.split != "val":
        raise ValueError(
            "Threshold calibration is strictly restricted to the 'val' split. "
            "Optimizing decision boundaries against the test split causes data leakage."
        )

    if request.target_model_id:
        model_rec = db.query(ModelRegistry).filter(ModelRegistry.model_id == request.target_model_id).first()
        if not model_rec:
            raise ValueError(f"Model '{request.target_model_id}' not found.")
    else:
        model_rec = get_active_model(db, model_type="embedding")

    model, device, transform = load_model_instance(model_rec)

    # 1. Build Reference Gallery for this model
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
                })
        except Exception:
            pass

    if not gallery:
        raise ValueError("Reference gallery is empty. Register SKU reference images before calibration.")

    # 2. Fetch Validation Images (strictly validation split)
    val_images = (
        db.query(ProductImage)
        .filter(
            ProductImage.split == "val",
            ProductImage.role.in_(["validation", "training", "query"]),
            ProductImage.role != "reference",
        )
        .all()
    )

    if not val_images:
        val_images = db.query(ProductImage).filter(ProductImage.role == "validation").all()

    # Precompute Query Embeddings and Matches once in memory for ultra-fast calibration sweep
    precomputed_queries = []
    total_known = 0
    total_unknown = 0

    for q_img in val_images:
        if not os.path.exists(q_img.file_path):
            continue
        gt_sku = q_img.sku.sku_code if q_img.sku else None
        is_unknown = (q_img.sku_id is None or gt_sku is None or gt_sku == "UNKNOWN")

        if is_unknown:
            total_unknown += 1
        else:
            total_known += 1

        try:
            with Image.open(q_img.file_path) as img:
                img = img.convert("RGB")
                tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_vec = model(tensor).squeeze().cpu().numpy().astype(np.float32)
                norm = np.linalg.norm(q_vec)
                if norm > 1e-6:
                    q_vec = q_vec / norm
        except Exception:
            continue

        sims = []
        for g in gallery:
            s = float(np.dot(q_vec, g["vector"]))
            sims.append((s, g["sku_code"]))

        sims.sort(key=lambda x: x[0], reverse=True)
        top1_sim, top1_code = sims[0] if sims else (0.0, "UNKNOWN")
        top2_sim = sims[1][0] if len(sims) > 1 else 0.0
        margin = round(top1_sim - top2_sim, 4)

        precomputed_queries.append({
            "is_unknown": is_unknown,
            "gt_sku": gt_sku,
            "top1_sim": top1_sim,
            "top1_code": top1_code,
            "margin": margin,
        })

    is_provisional = (total_unknown == 0 or total_known == 0)
    provisional_warning = None
    if total_unknown == 0:
        provisional_warning = (
            "Notice: Validation split contains 0 unknown negative samples. "
            "Calibration FAR is provisionally estimated at 0.0. "
            "Add or promote uncataloged negative samples to 'val' to enable data-driven FAR calibration."
        )

    # 3. Perform 2D Threshold Sweep
    sweep_results: List[ThresholdSweepPoint] = []

    # Generate similarity steps
    curr_sim = request.min_similarity
    sim_thresholds = []
    while curr_sim <= request.max_similarity + 1e-5:
        sim_thresholds.append(round(curr_sim, 3))
        curr_sim += request.similarity_step

    for sim_th in sim_thresholds:
        for m_th in request.margin_thresholds:
            correct_known = 0
            false_rej = 0
            false_acc = 0
            correct_unk = 0

            for q in precomputed_queries:
                is_accepted = (q["top1_sim"] >= sim_th) and (q["margin"] >= m_th)

                if q["is_unknown"]:
                    if is_accepted:
                        false_acc += 1
                    else:
                        correct_unk += 1
                else:
                    if is_accepted:
                        if q["top1_code"] == q["gt_sku"]:
                            correct_known += 1
                        else:
                            false_rej += 1
                    else:
                        false_rej += 1

            known_acc = round(correct_known / max(1, total_known), 4) if total_known > 0 else 0.0
            far = round(false_acc / max(1, total_unknown), 4) if total_unknown > 0 else 0.0
            frr = round(false_rej / max(1, total_known), 4) if total_known > 0 else 0.0
            rec = known_acc
            prec = round(correct_known / max(1, correct_known + false_acc), 4) if (correct_known + false_acc) > 0 else 0.0
            f1 = round(2 * (prec * rec) / max(1e-6, prec + rec), 4) if (prec + rec) > 0 else 0.0

            sweep_results.append(ThresholdSweepPoint(
                similarity_threshold=sim_th,
                margin_threshold=m_th,
                known_top1_accuracy=known_acc,
                unknown_far=far,
                known_frr=frr,
                precision=prec,
                recall=rec,
                f1_score=f1,
            ))

    # 4. Select Optimal Threshold according to Objective
    recommended_point = sweep_results[0] if sweep_results else None
    rationale = ""

    if request.objective == "min_far_with_recall_target":
        # Filter points satisfying recall target
        qualified = [p for p in sweep_results if p.recall >= request.min_recall_target]
        if qualified:
            # Sort by lowest FAR, then highest F1, then highest similarity threshold
            qualified.sort(key=lambda p: (p.unknown_far, -p.f1_score, -p.similarity_threshold))
            recommended_point = qualified[0]
            rationale = (
                f"Selected similarity={recommended_point.similarity_threshold:.2f}, "
                f"margin={recommended_point.margin_threshold:.2f} to minimize Unknown False Acceptance Rate "
                f"({recommended_point.unknown_far * 100:.1f}%) while maintaining Known Recall "
                f"({recommended_point.recall * 100:.1f}% >= target {request.min_recall_target * 100:.1f}%)."
            )
        else:
            # Fallback to highest recall available
            sweep_results_sorted = sorted(sweep_results, key=lambda p: (-p.recall, p.unknown_far))
            recommended_point = sweep_results_sorted[0]
            rationale = (
                f"No threshold combination achieved target recall {request.min_recall_target * 100:.1f}%. "
                f"Selected maximum achievable recall point ({recommended_point.recall * 100:.1f}%) "
                f"at similarity={recommended_point.similarity_threshold:.2f}, margin={recommended_point.margin_threshold:.2f}."
            )

    elif request.objective == "max_f1":
        sweep_results_sorted = sorted(sweep_results, key=lambda p: (-p.f1_score, p.unknown_far))
        recommended_point = sweep_results_sorted[0]
        rationale = (
            f"Selected similarity={recommended_point.similarity_threshold:.2f}, "
            f"margin={recommended_point.margin_threshold:.2f} to maximize Harmonic F1-Score "
            f"({recommended_point.f1_score:.4f}) with Precision={recommended_point.precision * 100:.1f}% and Recall={recommended_point.recall * 100:.1f}%."
        )

    elif request.objective == "balanced_eer":
        sweep_results_sorted = sorted(sweep_results, key=lambda p: (abs(p.unknown_far - p.known_frr), -p.f1_score))
        recommended_point = sweep_results_sorted[0]
        rationale = (
            f"Selected similarity={recommended_point.similarity_threshold:.2f}, "
            f"margin={recommended_point.margin_threshold:.2f} to achieve Equal Error Rate balance "
            f"(FAR={recommended_point.unknown_far * 100:.1f}%, FRR={recommended_point.known_frr * 100:.1f}%)."
        )

    # 5. Compute Per-SKU Performance at recommended threshold
    per_sku_acc: Dict[str, float] = {}
    if recommended_point:
        rec_sim = recommended_point.similarity_threshold
        rec_m = recommended_point.margin_threshold

        sku_counts: Dict[str, Dict[str, int]] = {}
        for q in precomputed_queries:
            if q["is_unknown"] or not q["gt_sku"]:
                continue
            sc = sku_counts.setdefault(q["gt_sku"], {"total": 0, "correct": 0})
            sc["total"] += 1
            if (q["top1_sim"] >= rec_sim) and (q["margin"] >= rec_m) and (q["top1_code"] == q["gt_sku"]):
                sc["correct"] += 1

        for code, sc in sku_counts.items():
            per_sku_acc[code] = round(sc["correct"] / max(1, sc["total"]), 4)

    return ThresholdCalibrationResponse(
        status="CALIBRATED" if not is_provisional else "PROVISIONAL_CALIBRATED",
        model_id=model_rec.model_id,
        split_evaluated="val",
        total_known_val_samples=total_known,
        total_unknown_val_samples=total_unknown,
        sweep_results=sweep_results,
        recommended_threshold=recommended_point.similarity_threshold if recommended_point else 0.60,
        recommended_margin=recommended_point.margin_threshold if recommended_point else 0.05,
        objective_used=request.objective,
        objective_rationale=rationale,
        is_provisional=is_provisional,
        provisional_warning=provisional_warning,
        per_sku_accuracy_at_recommended=per_sku_acc,
    )
