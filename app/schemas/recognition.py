from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class RecognitionCandidate(BaseModel):
    sku_id: int
    sku_code: str
    sku_name: str
    similarity_score: float = Field(..., description="Cosine similarity score [-1.0, 1.0]. Not a calibrated probability.")
    reference_image_id: Optional[int] = None
    reference_image_url: Optional[str] = None
    rank: int


class DetectedObject(BaseModel):
    box_id: int
    # Normalized coordinates [0..1]
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    # Absolute pixel coordinates
    px_x: int
    px_y: int
    px_width: int
    px_height: int

    detector_confidence: float = Field(..., description="Localization model confidence score [0.0, 1.0]")
    detector_class: str = Field(..., description="Class name from detection backbone")
    is_fallback: bool = Field(False, description="True if whole image or center frame fallback was used")

    crop_image_url: Optional[str] = None

    # Recognition results
    is_unknown: bool = Field(..., description="True if no candidate met the acceptance threshold or rules")
    recognition_status: str = Field("UNKNOWN", description="Decision status: KNOWN, UNKNOWN, or UNCLASSIFIED")
    predicted_sku: Optional[str] = Field(None, description="SKU code or UNKNOWN")
    predicted_sku_name: Optional[str] = None
    top_similarity_score: Optional[float] = Field(None, description="Highest candidate cosine similarity score")
    margin_threshold: Optional[float] = Field(None, description="Top-1 vs top-2 required margin threshold")
    margin_delta: Optional[float] = Field(None, description="Measured difference between top-1 and top-2 candidate similarity")
    rejection_reason: Optional[str] = None
    candidates: List[RecognitionCandidate] = []


class RecognitionResponse(BaseModel):
    query_image_name: str
    query_image_url: str
    image_width: int
    image_height: int
    detector_model: str
    embedding_model: str
    device_used: str
    detection_threshold: float
    similarity_threshold: float
    margin_threshold: Optional[float] = 0.0
    objects_detected_count: int
    has_fallback_detection: bool
    detected_objects: List[DetectedObject]
    duration_ms: float
    frame_seq: Optional[int] = None
    warnings: List[str] = []
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RecognitionLogSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    query_image_name: str
    num_detections: int
    final_status: str
    top_predicted_sku: Optional[str] = None
    top_similarity: Optional[float] = None
    detector_model: str
    embedding_model: str
    duration_ms: float
    created_at: datetime


class RecognitionStatusSummary(BaseModel):
    active_detector: str
    active_embedding_model: str
    embedding_dim: int
    gallery_reference_count: int
    registered_sku_count: int
    is_gallery_compatible: bool
    device_used: str
    default_detection_threshold: float
    default_similarity_threshold: float

