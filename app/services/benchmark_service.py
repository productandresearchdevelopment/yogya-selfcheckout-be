import os
import time
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image
import torch
from sqlalchemy.orm import Session

from app.models.sku import SKU
from app.models.image import ProductImage
from app.models.model_registry import ModelRegistry
from app.services.model_registry_service import (
    load_model_instance,
    activate_embedding_model,
    get_active_model,
)
from app.schemas.evaluation import (
    ComparativeBenchmarkRequest,
    ComparativeBenchmarkResponse,
    PerSkuBenchmarkDelta,
    PromotionSafeguardCheck,
    PromoteCandidateModelResponse,
)


def run_comparative_benchmark(
    db: Session,
    request: ComparativeBenchmarkRequest,
) -> ComparativeBenchmarkResponse:
    """
    Executes a strict, repeatable regression benchmark comparing a candidate fine-tuned model
    against the active/baseline model on identical queries and reference gallery.
    Evaluates Section 4 Requirement 7 Promotion Safeguards:
    1. Overall accuracy: candidate top1 >= baseline top1 - 0.02
    2. Unknown FAR: candidate FAR <= max(baseline FAR, 0.05)
    3. Per-SKU stability: no SKU accuracy regression > 0.15
    4. Dataset isolation: never evaluated on train split
    """
    if request.split not in ["val", "test"]:
        raise ValueError("Benchmark evaluation must run on 'val' or 'test'. Evaluating against 'train' causes data leakage.")

    candidate_rec = db.query(ModelRegistry).filter(ModelRegistry.model_id == request.candidate_model_id).first()
    if not candidate_rec:
        raise ValueError(f"Candidate model '{request.candidate_model_id}' not found in registry.")

    baseline_model_id = request.baseline_model_id or "mobilenetv3-pretrained-baseline"
    baseline_rec = db.query(ModelRegistry).filter(ModelRegistry.model_id == baseline_model_id).first()
    if not baseline_rec:
        # Fallback to active model if default baseline record not found
        baseline_rec = get_active_model(db, model_type="embedding")

    sim_th = request.similarity_threshold or 0.60
    margin_th = request.margin_threshold or 0.05

    # Fetch reference gallery images (shared ground truth)
    ref_images = (
        db.query(ProductImage)
        .join(SKU)
        .filter(ProductImage.role == "reference", SKU.is_active == True)
        .all()
    )

    # Fetch query evaluation images in specified split
    eval_images = (
        db.query(ProductImage)
        .filter(
            ProductImage.split == request.split,
            ProductImage.role.in_(["validation", "test", "training", "query"]),
            ProductImage.role != "reference",
        )
        .all()
    )

    if not eval_images:
        eval_images = db.query(ProductImage).filter(ProductImage.role == ("validation" if request.split == "val" else "test")).all()

    # 1. Evaluate Baseline Model
    baseline_metrics = _evaluate_single_model(baseline_rec, ref_images, eval_images, sim_th, margin_th)

    # 2. Evaluate Candidate Model
    candidate_metrics = _evaluate_single_model(candidate_rec, ref_images, eval_images, sim_th, margin_th)

    # 3. Compute Per-SKU Deltas and Regressions
    per_sku_breakdown: List[PerSkuBenchmarkDelta] = []
    severe_regressed_skus = []

    all_sku_codes = sorted(list(set(
        list(baseline_metrics["per_sku_accuracy"].keys()) +
        list(candidate_metrics["per_sku_accuracy"].keys())
    )))

    for code in all_sku_codes:
        base_acc = baseline_metrics["per_sku_accuracy"].get(code, 0.0)
        cand_acc = candidate_metrics["per_sku_accuracy"].get(code, 0.0)
        delta = round(cand_acc - base_acc, 4)
        has_regressed = (delta < -0.001)

        if delta < -0.15:
            severity = "SEVERE"
            severe_regressed_skus.append((code, delta))
        elif delta < -0.001:
            severity = "MINOR"
        else:
            severity = "NONE"

        per_sku_breakdown.append(PerSkuBenchmarkDelta(
            sku_code=code,
            baseline_accuracy=base_acc,
            candidate_accuracy=cand_acc,
            accuracy_delta=delta,
            has_regressed=has_regressed,
            regression_severity=severity,
        ))

    # 4. Check Promotion Safeguards
    safeguards: List[PromotionSafeguardCheck] = []
    rejection_reasons: List[str] = []

    # Safeguard 1: Overall Top-1 Accuracy Retention
    acc_diff = candidate_metrics["top1_accuracy"] - baseline_metrics["top1_accuracy"]
    sg1_pass = acc_diff >= -0.02
    safeguards.append(PromotionSafeguardCheck(
        safeguard_name="Overall Top-1 Accuracy Guard",
        status="PASS" if sg1_pass else "FAIL",
        required_criterion="Candidate Top-1 >= Baseline Top-1 - 2.0%",
        measured_value=f"Candidate: {candidate_metrics['top1_accuracy']*100:.1f}%, Baseline: {baseline_metrics['top1_accuracy']*100:.1f}% (Delta {acc_diff*100:+.1f}%)",
        passed=sg1_pass,
    ))
    if not sg1_pass:
        rejection_reasons.append(
            f"Overall Top-1 accuracy dropped by {-acc_diff*100:.1f}%, exceeding the 2.0% tolerance margin."
        )

    # Safeguard 2: Unknown False Acceptance Rate (FAR) Guard
    cand_far = candidate_metrics["unknown_false_acceptance_rate"]
    base_far = baseline_metrics["unknown_false_acceptance_rate"]
    max_allowed_far = max(base_far, 0.05)
    sg2_pass = cand_far <= max_allowed_far + 0.001
    safeguards.append(PromotionSafeguardCheck(
        safeguard_name="Unknown Object False Acceptance Rate (FAR) Guard",
        status="PASS" if sg2_pass else "FAIL",
        required_criterion=f"Candidate Unknown FAR <= {max_allowed_far*100:.1f}%",
        measured_value=f"Candidate FAR: {cand_far*100:.1f}%, Baseline FAR: {base_far*100:.1f}%",
        passed=sg2_pass,
    ))
    if not sg2_pass:
        rejection_reasons.append(
            f"Candidate allows excessive false acceptances on unknown objects: FAR={cand_far*100:.1f}% > {max_allowed_far*100:.1f}%."
        )

    # Safeguard 3: Per-SKU Stability Guard (No SKU Drops > 15%)
    sg3_pass = len(severe_regressed_skus) == 0
    safeguards.append(PromotionSafeguardCheck(
        safeguard_name="Per-SKU Severe Regression Guard",
        status="PASS" if sg3_pass else "FAIL",
        required_criterion="No individual SKU accuracy drop > 15.0%",
        measured_value=(
            f"{len(severe_regressed_skus)} SKU(s) regressed severely"
            if severe_regressed_skus
            else "All SKUs stable (max drop <= 15%)"
        ),
        passed=sg3_pass,
    ))
    if not sg3_pass:
        regressed_str = ", ".join([f"{c} (Delta {d*100:.1f}%)" for c, d in severe_regressed_skus])
        rejection_reasons.append(f"Severe accuracy regression detected for SKU(s): {regressed_str}.")

    # Safeguard 4: Split Isolation Guard
    sg4_pass = request.split in ["val", "test"]
    safeguards.append(PromotionSafeguardCheck(
        safeguard_name="Dataset Leakage Isolation Guard",
        status="PASS" if sg4_pass else "FAIL",
        required_criterion="Evaluation split is strictly independent ('val' or 'test')",
        measured_value=f"Evaluated on split '{request.split}'",
        passed=sg4_pass,
    ))

    eligible_for_promotion = all(s.passed for s in safeguards)

    # Comparison deltas summary
    comparison_deltas = {
        "top1_accuracy_delta": round(candidate_metrics["top1_accuracy"] - baseline_metrics["top1_accuracy"], 4),
        "top3_accuracy_delta": round(candidate_metrics["top3_accuracy"] - baseline_metrics["top3_accuracy"], 4),
        "unknown_far_delta": round(candidate_metrics["unknown_false_acceptance_rate"] - baseline_metrics["unknown_false_acceptance_rate"], 4),
        "known_frr_delta": round(candidate_metrics["known_false_rejection_rate"] - baseline_metrics["known_false_rejection_rate"], 4),
        "latency_delta_ms": round(candidate_metrics["average_latency_ms"] - baseline_metrics["average_latency_ms"], 2),
    }

    return ComparativeBenchmarkResponse(
        status="COMPLETED",
        candidate_model_id=candidate_rec.model_id,
        baseline_model_id=baseline_rec.model_id,
        split_evaluated=request.split,
        evaluated_samples_count=len(eval_images),
        similarity_threshold=sim_th,
        margin_threshold=margin_th,
        baseline_summary=baseline_metrics,
        candidate_summary=candidate_metrics,
        comparison_deltas=comparison_deltas,
        per_sku_breakdown=per_sku_breakdown,
        safeguards=safeguards,
        eligible_for_promotion=eligible_for_promotion,
        rejection_reasons=rejection_reasons,
    )


def execute_model_promotion_safeguarded(
    db: Session,
    candidate_model_id: str,
    force: bool = False,
) -> PromoteCandidateModelResponse:
    """
    Safely promotes a candidate model to active serving status.
    Strictly verifies all Section 4 safeguards.
    If any safeguard fails and force is False, rejects promotion and preserves active model & gallery.
    """
    # Run benchmark checks on validation split
    benchmark_req = ComparativeBenchmarkRequest(
        candidate_model_id=candidate_model_id,
        split="val",
    )
    bench_res = run_comparative_benchmark(db, benchmark_req)

    if not bench_res.eligible_for_promotion and not force:
        reasons_msg = "; ".join(bench_res.rejection_reasons)
        return PromoteCandidateModelResponse(
            promoted=False,
            status="BLOCKED_BY_SAFEGUARDS",
            model_id=candidate_model_id,
            previous_active_model_id=bench_res.baseline_model_id,
            migrated_gallery_count=0,
            message=f"Model promotion rejected by Section 4 Safeguards: {reasons_msg}. Active model preserved.",
            safeguards_status="FAIL",
        )

    # Perform atomic activation and gallery migration
    activation_res = activate_embedding_model(db, candidate_model_id)

    return PromoteCandidateModelResponse(
        promoted=True,
        status="PROMOTED",
        model_id=candidate_model_id,
        previous_active_model_id=activation_res.get("previous_model_id"),
        migrated_gallery_count=activation_res.get("migrated_vectors_count", 0),
        message=f"Model successfully promoted. {activation_res.get('message')}",
        safeguards_status="PASS" if bench_res.eligible_for_promotion else "FORCE_OVERRIDDEN",
    )


def _evaluate_single_model(
    model_rec: ModelRegistry,
    ref_images: List[ProductImage],
    eval_images: List[ProductImage],
    similarity_threshold: float,
    margin_threshold: float,
) -> Dict[str, Any]:
    """Runs evaluation on a single model record against reference gallery and queries."""
    model, device, transform = load_model_instance(model_rec)

    # Build gallery
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
        return {
            "model_id": model_rec.model_id,
            "top1_accuracy": 0.0,
            "top3_accuracy": 0.0,
            "unknown_false_acceptance_rate": 0.0,
            "known_false_rejection_rate": 0.0,
            "average_latency_ms": 0.0,
            "per_sku_accuracy": {},
        }

    latencies = []
    total_known = 0
    correct_top1 = 0
    correct_top3 = 0
    false_rejections = 0
    total_unknown = 0
    false_acceptances = 0
    per_sku_counts: Dict[str, Dict[str, int]] = {}

    for q_img in eval_images:
        if not os.path.exists(q_img.file_path):
            continue

        gt_sku = q_img.sku.sku_code if q_img.sku else None
        is_unknown = (q_img.sku_id is None or gt_sku is None or gt_sku == "UNKNOWN")

        t0 = time.time()
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
        latencies.append((time.time() - t0) * 1000)

        sims = []
        for g in gallery:
            s = float(np.dot(q_vec, g["vector"]))
            sims.append((s, g["sku_code"]))

        sims.sort(key=lambda x: x[0], reverse=True)
        top1_sim, top1_code = sims[0] if sims else (0.0, "UNKNOWN")
        top2_sim = sims[1][0] if len(sims) > 1 else 0.0
        margin = round(top1_sim - top2_sim, 4)

        top3_codes = [x[1] for x in sims[:3]]

        is_accepted = (top1_sim >= similarity_threshold) and (margin >= margin_threshold)

        if is_unknown:
            total_unknown += 1
            if is_accepted:
                false_acceptances += 1
        else:
            total_known += 1
            sc = per_sku_counts.setdefault(gt_sku, {"total": 0, "correct": 0})
            sc["total"] += 1

            if is_accepted:
                if top1_code == gt_sku:
                    correct_top1 += 1
                    sc["correct"] += 1
                if gt_sku in top3_codes:
                    correct_top3 += 1
            else:
                false_rejections += 1

    return {
        "model_id": model_rec.model_id,
        "architecture": model_rec.architecture,
        "embedding_dim": model_rec.embedding_dim,
        "top1_accuracy": round(correct_top1 / max(1, total_known), 4) if total_known > 0 else 0.0,
        "top3_accuracy": round(correct_top3 / max(1, total_known), 4) if total_known > 0 else 0.0,
        "unknown_false_acceptance_rate": round(false_acceptances / max(1, total_unknown), 4) if total_unknown > 0 else 0.0,
        "known_false_rejection_rate": round(false_rejections / max(1, total_known), 4) if total_known > 0 else 0.0,
        "average_latency_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
        "total_known_queries": total_known,
        "total_unknown_queries": total_unknown,
        "per_sku_accuracy": {
            k: round(v["correct"] / max(1, v["total"]), 4) for k, v in per_sku_counts.items()
        },
    }
