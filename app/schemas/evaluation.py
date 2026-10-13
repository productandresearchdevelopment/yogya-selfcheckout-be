from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# HARD NEGATIVES SCHEMAS
# ---------------------------------------------------------------------------

class HardNegativeCreate(BaseModel):
    sample_type: str = Field("manual_upload", description="inference_log, manual_upload, synthetic_eval, webcam_review")
    image_path: str
    crop_path: Optional[str] = None
    predicted_sku: Optional[str] = None
    predicted_sku_name: Optional[str] = None
    ground_truth_sku: Optional[str] = None
    ground_truth_label: str = Field("UNKNOWN", description="'UNKNOWN' or registered SKU code")
    detector_confidence: Optional[float] = None
    similarity_score: Optional[float] = None
    margin_delta: Optional[float] = None
    top_candidates_json: Optional[str] = None
    model_version: str
    error_category: str = Field("FALSE_POSITIVE_UNKNOWN_AS_KNOWN")
    review_status: str = Field("pending")
    reviewer_notes: Optional[str] = None


class HardNegativeReviewUpdate(BaseModel):
    review_status: str = Field(..., description="'confirmed', 'corrected', or 'rejected'")
    ground_truth_sku: Optional[str] = None
    ground_truth_label: Optional[str] = Field("UNKNOWN")
    reviewer_notes: Optional[str] = None
    error_category: Optional[str] = None


class HardNegativeRecordResponse(BaseModel):
    id: int
    sample_type: str
    image_path: str
    crop_path: Optional[str] = None
    crop_url: Optional[str] = None
    predicted_sku: Optional[str] = None
    predicted_sku_name: Optional[str] = None
    ground_truth_sku: Optional[str] = None
    ground_truth_label: str
    detector_confidence: Optional[float] = None
    similarity_score: Optional[float] = None
    margin_delta: Optional[float] = None
    top_candidates: Optional[List[Dict[str, Any]]] = None
    model_version: str
    error_category: str
    review_status: str
    reviewer_notes: Optional[str] = None
    is_mined_for_training: bool
    dataset_split_assigned: Optional[str] = None
    created_at: str
    reviewed_at: Optional[str] = None

    model_config = {"from_attributes": True}


class HardNegativeMineRequest(BaseModel):
    min_confidence: float = Field(0.2, ge=0.0, le=1.0)
    min_similarity: float = Field(0.35, ge=0.0, le=1.0)
    max_similarity: float = Field(0.99, ge=0.0, le=1.0)
    include_unknowns_only: bool = Field(False)
    limit: int = Field(50, ge=1, le=200)


class HardNegativeMineResponse(BaseModel):
    mined_count: int
    message: str
    records: List[HardNegativeRecordResponse]


class PromoteHardNegativesRequest(BaseModel):
    record_ids: List[int]
    target_split: str = Field("train", description="'train' or 'val' (never 'test')")
    as_sku_id: Optional[int] = Field(None, description="SKU ID to assign if known product, or None for UNKNOWN negative sample")


class PromoteHardNegativesResponse(BaseModel):
    promoted_count: int
    target_split: str
    created_image_ids: List[int]
    message: str


# ---------------------------------------------------------------------------
# ERROR ANALYSIS SCHEMAS
# ---------------------------------------------------------------------------

class PerSkuMetric(BaseModel):
    sku_id: Optional[int] = None
    sku_code: str
    sku_name: str
    total_queries: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    mean_similarity: float


class ConfusionMatrixData(BaseModel):
    labels: List[str]
    matrix: List[List[int]]


class ErrorExampleRecord(BaseModel):
    query_id: int
    filename: str
    image_url: Optional[str] = None
    ground_truth_label: str
    predicted_label: str
    similarity_score: float
    margin_delta: Optional[float] = None
    error_category: str
    detector_confidence: Optional[float] = None
    reason: str


class ErrorAnalysisResponse(BaseModel):
    status: str
    model_id: str
    split_evaluated: str
    similarity_threshold: float
    margin_threshold: float
    total_samples: int
    total_known_samples: int
    total_unknown_samples: int
    category_breakdown: Dict[str, int]
    per_sku_metrics: List[PerSkuMetric]
    confusion_matrix: ConfusionMatrixData
    representative_errors: List[ErrorExampleRecord]
    detector_metrics_separated: Dict[str, Any]
    summary: Dict[str, Any]


# ---------------------------------------------------------------------------
# THRESHOLD CALIBRATION SCHEMAS
# ---------------------------------------------------------------------------

class ThresholdSweepPoint(BaseModel):
    similarity_threshold: float
    margin_threshold: float
    known_top1_accuracy: float
    unknown_far: float
    known_frr: float
    precision: float
    recall: float
    f1_score: float


class ThresholdCalibrationRequest(BaseModel):
    target_model_id: Optional[str] = None
    split: str = Field("val", description="Strictly 'val' - held-out test split is isolated")
    min_similarity: float = Field(0.35, ge=0.1, le=0.9)
    max_similarity: float = Field(0.85, ge=0.2, le=0.99)
    similarity_step: float = Field(0.05, ge=0.01, le=0.2)
    margin_thresholds: List[float] = Field(default_factory=lambda: [0.0, 0.05, 0.10, 0.15])
    objective: str = Field("min_far_with_recall_target", description="'min_far_with_recall_target', 'max_f1', or 'balanced_eer'")
    min_recall_target: float = Field(0.85, ge=0.5, le=1.0)


class ThresholdCalibrationResponse(BaseModel):
    status: str
    model_id: str
    split_evaluated: str
    total_known_val_samples: int
    total_unknown_val_samples: int
    sweep_results: List[ThresholdSweepPoint]
    recommended_threshold: float
    recommended_margin: float
    objective_used: str
    objective_rationale: str
    is_provisional: bool
    provisional_warning: Optional[str] = None
    per_sku_accuracy_at_recommended: Dict[str, float]


# ---------------------------------------------------------------------------
# REGRESSION BENCHMARK & PROMOTION SAFEGUARDS SCHEMAS
# ---------------------------------------------------------------------------

class PerSkuBenchmarkDelta(BaseModel):
    sku_code: str
    baseline_accuracy: float
    candidate_accuracy: float
    accuracy_delta: float
    has_regressed: bool
    regression_severity: str  # 'NONE', 'MINOR', 'SEVERE'


class PromotionSafeguardCheck(BaseModel):
    safeguard_name: str
    status: str  # 'PASS' or 'FAIL'
    required_criterion: str
    measured_value: str
    passed: bool


class ComparativeBenchmarkRequest(BaseModel):
    candidate_model_id: str
    baseline_model_id: Optional[str] = Field("mobilenetv3-pretrained-baseline")
    split: str = Field("val", description="'val' or 'test'")
    similarity_threshold: Optional[float] = Field(0.60, ge=0.1, le=0.99)
    margin_threshold: Optional[float] = Field(0.05, ge=0.0, le=0.5)


class ComparativeBenchmarkResponse(BaseModel):
    status: str
    candidate_model_id: str
    baseline_model_id: str
    split_evaluated: str
    evaluated_samples_count: int
    similarity_threshold: float
    margin_threshold: float
    baseline_summary: Dict[str, Any]
    candidate_summary: Dict[str, Any]
    comparison_deltas: Dict[str, Any]
    per_sku_breakdown: List[PerSkuBenchmarkDelta]
    safeguards: List[PromotionSafeguardCheck]
    eligible_for_promotion: bool
    rejection_reasons: List[str]


class PromoteCandidateModelRequest(BaseModel):
    candidate_model_id: str
    force: bool = Field(False, description="Override safeguards if explicitly approved by human supervisor")


class PromoteCandidateModelResponse(BaseModel):
    promoted: bool
    status: str
    model_id: str
    previous_active_model_id: Optional[str] = None
    migrated_gallery_count: int
    message: str
    safeguards_status: str
