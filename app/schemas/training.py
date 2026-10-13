from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class DatasetReadinessSummary(BaseModel):
    total_skus: int
    active_skus: int
    total_images: int
    reference_images: int
    training_images: int
    validation_images: int
    test_images: int
    unassigned_images: int
    distinct_training_identities: int
    skus_with_multiple_images: int
    skus_with_zero_images: int
    hard_negative_unknown_count: int
    corrupt_images_count: int
    duplicate_images_count: int
    blurry_images_count: int
    annotated_bounding_boxes: int


class SkuReadinessDetail(BaseModel):
    sku_id: int
    sku_code: str
    sku_name: str
    total_images: int
    reference_images: int
    training_images: int
    val_images: int
    test_images: int
    has_sufficient_pairs: bool


class DatasetReadinessResponse(BaseModel):
    status: str
    is_ready: bool
    summary: DatasetReadinessSummary
    blockers: List[str]
    recommendations: List[str]
    sku_details: List[SkuReadinessDetail]


class StartTrainingRequest(BaseModel):
    epochs: int = Field(default=5, ge=1, le=50)
    embedding_dim: int = Field(default=512, ge=128, le=1024)
    learning_rate: float = Field(default=0.0001, ge=1e-6, le=0.01)
    margin: float = Field(default=0.2, ge=0.01, le=1.0)
    p_classes: int = Field(default=2, ge=2, le=8)
    k_instances: int = Field(default=2, ge=2, le=8)


class TrainingJobResponse(BaseModel):
    job_id: str
    model_id: str
    status: str
    total_epochs: int
    current_epoch: int
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None
    history: List[Dict[str, Any]] = []
    logs: Optional[str] = None
    error_message: Optional[str] = None
    checkpoint_path: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    created_at: str


class ModelRegistryResponse(BaseModel):
    id: int
    model_id: str
    model_type: str
    architecture: str
    pretrained_source: str
    weight_version: str
    embedding_dim: int
    checkpoint_path: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    status: str
    is_active: bool
    license: str
    created_at: str


class ModelActivationResponse(BaseModel):
    status: str
    model_id: str
    previous_model_id: Optional[str] = None
    migrated_vectors_count: int
    embedding_dim: int
    message: str


class DetectorEvaluationResponse(BaseModel):
    status: str
    detector_model: Optional[str] = None
    confidence_threshold: float
    metrics: Dict[str, Any]
    eval_duration_ms: float
    message: Optional[str] = None


class EmbeddingEvaluationResponse(BaseModel):
    status: str
    evaluated_images_count: Optional[int] = None
    gallery_reference_count: Optional[int] = None
    similarity_threshold: Optional[float] = None
    baseline_model: Optional[Dict[str, Any]] = None
    active_model: Optional[Dict[str, Any]] = None
    comparison_delta: Optional[Dict[str, Any]] = None
    threshold_calibration: Optional[List[Dict[str, Any]]] = None
    message: Optional[str] = None
