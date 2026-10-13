import os
import json
import shutil
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from PIL import Image

from app.core.config import STORAGE_DIR, IMAGES_DIR
from app.models.hard_negative import HardNegativeRecord
from app.models.recognition_log import RecognitionLog
from app.models.sku import SKU
from app.models.image import ProductImage
from app.schemas.evaluation import (
    HardNegativeCreate,
    HardNegativeReviewUpdate,
    HardNegativeRecordResponse,
)


def list_hard_negatives(
    db: Session,
    status: Optional[str] = None,
    error_category: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[HardNegativeRecord]:
    """Retrieves hard negative records with optional filtering by review status or error category."""
    query = db.query(HardNegativeRecord)
    if status and status != "all":
        query = query.filter(HardNegativeRecord.review_status == status)
    if error_category and error_category != "all":
        query = query.filter(HardNegativeRecord.error_category == error_category)
    return query.order_by(HardNegativeRecord.created_at.desc()).offset(offset).limit(limit).all()


def get_hard_negative_by_id(db: Session, record_id: int) -> Optional[HardNegativeRecord]:
    """Retrieves a single hard negative record by ID."""
    return db.query(HardNegativeRecord).filter(HardNegativeRecord.id == record_id).first()


def create_hard_negative_record(db: Session, data: HardNegativeCreate) -> HardNegativeRecord:
    """Manually creates a new hard negative record."""
    record = HardNegativeRecord(
        sample_type=data.sample_type,
        image_path=data.image_path,
        crop_path=data.crop_path,
        predicted_sku=data.predicted_sku,
        predicted_sku_name=data.predicted_sku_name,
        ground_truth_sku=data.ground_truth_sku,
        ground_truth_label=data.ground_truth_label,
        detector_confidence=data.detector_confidence,
        similarity_score=data.similarity_score,
        margin_delta=data.margin_delta,
        top_candidates_json=data.top_candidates_json,
        model_version=data.model_version,
        error_category=data.error_category,
        review_status=data.review_status,
        reviewer_notes=data.reviewer_notes,
        is_mined_for_training=False,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def review_hard_negative_record(
    db: Session,
    record_id: int,
    update_data: HardNegativeReviewUpdate,
) -> HardNegativeRecord:
    """Updates review status, ground truth labels, and notes for a hard negative record."""
    record = get_hard_negative_by_id(db, record_id)
    if not record:
        raise ValueError(f"Hard negative record #{record_id} not found")

    record.review_status = update_data.review_status
    if update_data.ground_truth_sku is not None:
        record.ground_truth_sku = update_data.ground_truth_sku
    if update_data.ground_truth_label is not None:
        record.ground_truth_label = update_data.ground_truth_label
    if update_data.reviewer_notes is not None:
        record.reviewer_notes = update_data.reviewer_notes
    if update_data.error_category is not None:
        record.error_category = update_data.error_category

    record.reviewed_at = datetime.utcnow()
    db.commit()
    db.refresh(record)
    return record


def delete_hard_negative_record(db: Session, record_id: int) -> bool:
    """Deletes a hard negative record from the review queue."""
    record = get_hard_negative_by_id(db, record_id)
    if not record:
        return False
    db.delete(record)
    db.commit()
    return True


def mine_hard_negatives_from_logs(
    db: Session,
    min_confidence: float = 0.20,
    min_similarity: float = 0.35,
    max_similarity: float = 0.99,
    include_unknowns_only: bool = False,
    limit: int = 50,
) -> List[HardNegativeRecord]:
    """
    Mines difficult inference observations from recognition_logs.
    Extracts crops where similarity was ambiguous or unknown objects received high similarity.
    Records provenance and top candidates for human review.
    """
    logs = (
        db.query(RecognitionLog)
        .order_by(RecognitionLog.created_at.desc())
        .limit(200)
        .all()
    )

    mined_records: List[HardNegativeRecord] = []

    for log in logs:
        if len(mined_records) >= limit:
            break

        if not log.results_json:
            continue

        try:
            boxes = json.loads(log.results_json)
        except Exception:
            continue

        for box in boxes:
            if len(mined_records) >= limit:
                break

            top_sim = float(box.get("top_similarity_score") or log.top_similarity or 0.0)
            det_conf = float(box.get("detector_confidence") or 0.0)
            is_unknown = box.get("is_unknown", False) or (log.final_status == "UNKNOWN")
            candidates = box.get("candidates") or []

            # Check criteria for hard negative candidate
            # 1. Ambiguous similarity or near threshold
            # 2. Classified as unknown despite detection
            # 3. High similarity on suspected unknown object
            if include_unknowns_only and not is_unknown:
                continue

            if top_sim < min_similarity and not is_unknown:
                continue

            # Calculate margin delta (top1 - top2)
            margin_delta = 0.0
            if len(candidates) >= 2:
                margin_delta = round(candidates[0].get("similarity_score", 0.0) - candidates[1].get("similarity_score", 0.0), 4)

            # Determine initial error category
            if is_unknown and top_sim >= 0.50:
                err_cat = "FALSE_POSITIVE_UNKNOWN_AS_KNOWN"
            elif is_unknown:
                err_cat = "FALSE_REJECTION_KNOWN_AS_UNKNOWN"
            elif margin_delta < 0.05 and margin_delta > 0.0:
                err_cat = "WRONG_SKU_MISCLASSIFICATION"
            else:
                err_cat = "FALSE_POSITIVE_UNKNOWN_AS_KNOWN"

            # Check if this crop or log is already recorded to prevent duplicates
            crop_url = box.get("crop_image_url")
            existing = (
                db.query(HardNegativeRecord)
                .filter(
                    HardNegativeRecord.image_path == log.query_image_path,
                    HardNegativeRecord.predicted_sku == (box.get("predicted_sku") or log.top_predicted_sku),
                )
                .first()
            )
            if existing:
                continue

            rec = HardNegativeRecord(
                sample_type="inference_log",
                image_path=log.query_image_path or "",
                crop_path=crop_url,
                predicted_sku=box.get("predicted_sku") or log.top_predicted_sku,
                predicted_sku_name=box.get("predicted_sku_name"),
                ground_truth_sku=None,
                ground_truth_label="UNKNOWN",
                detector_confidence=det_conf,
                similarity_score=top_sim,
                margin_delta=margin_delta,
                top_candidates_json=json.dumps(candidates),
                model_version=f"{log.embedding_model or 'mobilenetv3'} | {log.detector_model or 'fasterrcnn'}",
                error_category=err_cat,
                review_status="pending",
                reviewer_notes=f"Auto-mined from Recognition Log #{log.id}. Status: {log.final_status}",
                is_mined_for_training=False,
            )
            db.add(rec)
            mined_records.append(rec)

    if mined_records:
        db.commit()
        for r in mined_records:
            db.refresh(r)

    return mined_records


def promote_hard_negatives_to_dataset(
    db: Session,
    record_ids: List[int],
    target_split: str = "train",
    as_sku_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Safely promotes reviewed hard-negative records into the dataset.
    Strictly prohibits promoting into 'test' to prevent test set data leakage.
    Creates versioned ProductImage records with full provenance metadata.
    """
    if target_split not in ["train", "val"]:
        raise ValueError("Target split must be 'train' or 'val'. Promoting into 'test' is prohibited to prevent data leakage.")

    records = (
        db.query(HardNegativeRecord)
        .filter(HardNegativeRecord.id.in_(record_ids))
        .all()
    )

    created_image_ids = []

    for rec in records:
        if rec.review_status not in ["confirmed", "corrected"]:
            # Only reviewed and approved records may be promoted
            continue

        # Determine source file
        src_path = rec.image_path
        if not os.path.exists(src_path):
            continue

        # Copy to dataset originals directory with unique name to guarantee immutability
        filename = f"hardneg_{rec.id}_{os.path.basename(src_path)}"
        dst_path = IMAGES_DIR / "original" / filename
        os.makedirs(dst_path.parent, exist_ok=True)
        shutil.copy2(src_path, dst_path)

        # File hash and dimensions
        with open(dst_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        with Image.open(dst_path) as img:
            width, height = img.size

        # Determine SKU ID
        final_sku_id = as_sku_id
        if final_sku_id is None and rec.ground_truth_sku:
            sku_obj = db.query(SKU).filter(SKU.sku_code == rec.ground_truth_sku).first()
            if sku_obj:
                final_sku_id = sku_obj.id

        role = "training" if target_split == "train" else "validation"
        provenance = (
            f"Promoted from Hard Negative Record #{rec.id}. "
            f"Model Version: {rec.model_version}. "
            f"Original Error Category: {rec.error_category}. "
            f"Reviewer Notes: {rec.reviewer_notes or 'Confirmed'}"
        )

        prod_image = ProductImage(
            sku_id=final_sku_id,
            role=role,
            split=target_split,
            filename=filename,
            file_path=str(dst_path),
            file_hash=file_hash,
            file_size_bytes=os.path.getsize(dst_path),
            width=width,
            height=height,
            mime_type="image/jpeg",
            is_valid=True,
            validation_notes=provenance,
        )
        db.add(prod_image)
        db.flush()
        created_image_ids.append(prod_image.id)

        # Mark hard negative record as promoted
        rec.is_mined_for_training = True
        rec.dataset_split_assigned = target_split

    db.commit()

    return {
        "promoted_count": len(created_image_ids),
        "target_split": target_split,
        "created_image_ids": created_image_ids,
        "message": f"Successfully promoted {len(created_image_ids)} reviewed hard-negative records into {target_split} dataset split.",
    }


def format_hard_negative_response(rec: HardNegativeRecord) -> HardNegativeRecordResponse:
    """Formats a database HardNegativeRecord into a response schema."""
    top_candidates = []
    if rec.top_candidates_json:
        try:
            top_candidates = json.loads(rec.top_candidates_json)
        except Exception:
            pass

    return HardNegativeRecordResponse(
        id=rec.id,
        sample_type=rec.sample_type,
        image_path=rec.image_path,
        crop_path=rec.crop_path,
        crop_url=rec.crop_path if rec.crop_path and rec.crop_path.startswith("/") else None,
        predicted_sku=rec.predicted_sku,
        predicted_sku_name=rec.predicted_sku_name,
        ground_truth_sku=rec.ground_truth_sku,
        ground_truth_label=rec.ground_truth_label,
        detector_confidence=rec.detector_confidence,
        similarity_score=rec.similarity_score,
        margin_delta=rec.margin_delta,
        top_candidates=top_candidates,
        model_version=rec.model_version,
        error_category=rec.error_category,
        review_status=rec.review_status,
        reviewer_notes=rec.reviewer_notes,
        is_mined_for_training=rec.is_mined_for_training,
        dataset_split_assigned=rec.dataset_split_assigned,
        created_at=str(rec.created_at),
        reviewed_at=str(rec.reviewed_at) if rec.reviewed_at else None,
    )
