from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text
from app.core.database import Base


class HardNegativeRecord(Base):
    __tablename__ = "hard_negatives"

    id = Column(Integer, primary_key=True, index=True)
    sample_type = Column(String(50), default="inference_log", nullable=False)  # 'inference_log', 'manual_upload', 'synthetic_eval', 'webcam_review'
    image_path = Column(String(512), nullable=False)
    crop_path = Column(String(512), nullable=True)
    predicted_sku = Column(String(100), nullable=True)
    predicted_sku_name = Column(String(255), nullable=True)
    ground_truth_sku = Column(String(100), nullable=True)  # None if UNKNOWN non-catalog object
    ground_truth_label = Column(String(100), default="UNKNOWN", nullable=False)  # 'UNKNOWN' or SKU code
    detector_confidence = Column(Float, nullable=True)
    similarity_score = Column(Float, nullable=True)
    margin_delta = Column(Float, nullable=True)
    top_candidates_json = Column(Text, nullable=True)
    model_version = Column(String(100), nullable=False)
    error_category = Column(
        String(100),
        default="FALSE_POSITIVE_UNKNOWN_AS_KNOWN",
        nullable=False,
    )  # 'FALSE_POSITIVE_UNKNOWN_AS_KNOWN', 'WRONG_SKU_MISCLASSIFICATION', 'FALSE_REJECTION_KNOWN_AS_UNKNOWN', 'DETECTOR_POOR_CROP', 'DETECTOR_MISSED_OBJECT', 'LOW_QUALITY_OCCLUDED', 'CORRECT_KNOWN', 'CORRECT_UNKNOWN'
    review_status = Column(String(50), default="pending", nullable=False)  # 'pending', 'confirmed', 'corrected', 'rejected'
    reviewer_notes = Column(Text, nullable=True)
    is_mined_for_training = Column(Boolean, default=False, nullable=False)
    dataset_split_assigned = Column(String(50), nullable=True)  # 'train', 'val', 'test'
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    reviewed_at = Column(DateTime, nullable=True)
