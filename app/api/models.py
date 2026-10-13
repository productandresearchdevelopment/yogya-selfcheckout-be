import json
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.model_registry import ModelRegistry
from app.schemas.training import (
    ModelRegistryResponse,
    ModelActivationResponse,
    DetectorEvaluationResponse,
    EmbeddingEvaluationResponse,
)
from app.services.model_registry_service import (
    list_registered_models,
    get_active_model,
    activate_embedding_model,
    ensure_baseline_models_registered,
)
from app.services.evaluation_service import (
    evaluate_detector_localization,
    evaluate_embedding_model_comparative,
)

router = APIRouter(prefix="/models", tags=["Model Registry & Evaluation"])


@router.get("/", response_model=List[ModelRegistryResponse])
def list_models(
    model_type: Optional[str] = Query(None, description="Filter by 'embedding' or 'detector'"),
    db: Session = Depends(get_db),
):
    """Lists all registered pretrained and fine-tuned models in the versioned registry."""
    ensure_baseline_models_registered(db)
    models = list_registered_models(db, model_type=model_type)
    return [_format_model_response(m) for m in models]


@router.get("/active", response_model=ModelRegistryResponse)
def get_current_active_model(
    model_type: str = Query("embedding", description="'embedding' or 'detector'"),
    db: Session = Depends(get_db),
):
    """Fetches the currently active model version for inference."""
    active = get_active_model(db, model_type=model_type)
    return _format_model_response(active)


@router.post("/{model_id}/activate", response_model=ModelActivationResponse)
def activate_model_endpoint(model_id: str, db: Session = Depends(get_db)):
    """
    Atomically activates a model checkpoint and regenerates reference gallery embeddings.
    Adheres strictly to Requirement 8: validates checkpoint first, regenerates in memory,
    executes atomic DB update, and rolls back if any step fails.
    """
    try:
        result = activate_embedding_model(db, model_id)
        return ModelActivationResponse(
            status=result["status"],
            model_id=result["model_id"],
            previous_model_id=result.get("previous_model_id"),
            migrated_vectors_count=result.get("migrated_vectors_count", 0),
            embedding_dim=result.get("embedding_dim", 0),
            message=result["message"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Activation and gallery migration failed: {str(e)}",
        )


@router.post("/evaluate/detector", response_model=DetectorEvaluationResponse)
def run_detector_evaluation(
    confidence_threshold: float = Query(0.35, ge=0.05, le=0.95),
    db: Session = Depends(get_db),
):
    """
    Evaluates object detector localization accuracy against ground-truth bounding box annotations.
    Separated from embedding recognition per Requirement 5.
    """
    res = evaluate_detector_localization(db, confidence_threshold=confidence_threshold)
    return DetectorEvaluationResponse(
        status=res["status"],
        detector_model=res.get("detector_model"),
        confidence_threshold=confidence_threshold,
        metrics=res["metrics"],
        eval_duration_ms=res["eval_duration_ms"],
        message=res.get("message"),
    )


@router.post("/evaluate/embedding", response_model=EmbeddingEvaluationResponse)
def run_embedding_evaluation(
    target_model_id: Optional[str] = Query(None, description="Optional target model ID (defaults to active)"),
    similarity_threshold: float = Query(0.60, ge=0.1, le=0.99),
    db: Session = Depends(get_db),
):
    """
    Runs comparative evaluation between Pretrained Baseline and Target/Fine-Tuned model
    on the exact same test queries, reporting Top-1/Top-3, FAR, FRR, and hard negative performance.
    """
    res = evaluate_embedding_model_comparative(
        db, target_model_id=target_model_id, similarity_threshold=similarity_threshold
    )
    return EmbeddingEvaluationResponse(
        status=res["status"],
        evaluated_images_count=res.get("evaluated_images_count"),
        gallery_reference_count=res.get("gallery_reference_count"),
        similarity_threshold=res.get("similarity_threshold"),
        baseline_model=res.get("baseline_model"),
        active_model=res.get("active_model"),
        comparison_delta=res.get("comparison_delta"),
        threshold_calibration=res.get("threshold_calibration"),
        message=res.get("message"),
    )


def _format_model_response(m: ModelRegistry) -> ModelRegistryResponse:
    hp = {}
    if m.hyperparameters_json:
        try:
            hp = json.loads(m.hyperparameters_json)
        except Exception:
            pass

    metrics = {}
    if m.metrics_json:
        try:
            metrics = json.loads(m.metrics_json)
        except Exception:
            pass

    return ModelRegistryResponse(
        id=m.id,
        model_id=m.model_id,
        model_type=m.model_type,
        architecture=m.architecture,
        pretrained_source=m.pretrained_source,
        weight_version=m.weight_version,
        embedding_dim=m.embedding_dim,
        checkpoint_path=m.checkpoint_path,
        hyperparameters=hp,
        metrics=metrics,
        status=m.status,
        is_active=m.is_active,
        license=m.license,
        created_at=str(m.created_at),
    )
