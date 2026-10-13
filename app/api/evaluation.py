from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.evaluation import (
    HardNegativeCreate,
    HardNegativeReviewUpdate,
    HardNegativeRecordResponse,
    HardNegativeMineRequest,
    HardNegativeMineResponse,
    PromoteHardNegativesRequest,
    PromoteHardNegativesResponse,
    ErrorAnalysisResponse,
    ThresholdCalibrationRequest,
    ThresholdCalibrationResponse,
    ComparativeBenchmarkRequest,
    ComparativeBenchmarkResponse,
    PromoteCandidateModelRequest,
    PromoteCandidateModelResponse,
)
from app.services.hard_negative_service import (
    list_hard_negatives,
    get_hard_negative_by_id,
    create_hard_negative_record,
    review_hard_negative_record,
    delete_hard_negative_record,
    mine_hard_negatives_from_logs,
    promote_hard_negatives_to_dataset,
    format_hard_negative_response,
)
from app.services.error_analysis_service import run_error_analysis
from app.services.threshold_calibration_service import run_threshold_calibration
from app.services.benchmark_service import (
    run_comparative_benchmark,
    execute_model_promotion_safeguarded,
)

router = APIRouter(prefix="/evaluation", tags=["Model Evaluation & Hard-Negative Mining"])


# ---------------------------------------------------------------------------
# 1. HARD-NEGATIVE MINING & REVIEW QUEUE ENDPOINTS
# ---------------------------------------------------------------------------

@router.get("/hard-negatives", response_model=List[HardNegativeRecordResponse])
def get_hard_negatives(
    status: Optional[str] = Query(None, description="Filter by status: 'pending', 'confirmed', 'corrected', 'rejected', or 'all'"),
    category: Optional[str] = Query(None, description="Filter by error category"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """Lists hard-negative review queue records with optional filtering."""
    records = list_hard_negatives(db, status=status, error_category=category, limit=limit, offset=offset)
    return [format_hard_negative_response(r) for r in records]


@router.post("/hard-negatives", response_model=HardNegativeRecordResponse)
def create_hard_negative(data: HardNegativeCreate, db: Session = Depends(get_db)):
    """Manually creates a new hard-negative sample."""
    record = create_hard_negative_record(db, data)
    return format_hard_negative_response(record)


@router.get("/hard-negatives/{record_id}", response_model=HardNegativeRecordResponse)
def get_hard_negative(record_id: int, db: Session = Depends(get_db)):
    """Retrieves a single hard-negative record by ID."""
    rec = get_hard_negative_by_id(db, record_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Hard negative record not found")
    return format_hard_negative_response(rec)


@router.put("/hard-negatives/{record_id}/review", response_model=HardNegativeRecordResponse)
def update_hard_negative_review(
    record_id: int,
    data: HardNegativeReviewUpdate,
    db: Session = Depends(get_db),
):
    """Reviews, corrects, confirms, or rejects a hard-negative sample."""
    try:
        updated = review_hard_negative_record(db, record_id, data)
        return format_hard_negative_response(updated)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/hard-negatives/{record_id}")
def delete_hard_negative(record_id: int, db: Session = Depends(get_db)):
    """Removes a hard-negative record from the review queue."""
    success = delete_hard_negative_record(db, record_id)
    if not success:
        raise HTTPException(status_code=404, detail="Hard negative record not found")
    return {"status": "DELETED", "id": record_id}


@router.post("/hard-negatives/mine-from-logs", response_model=HardNegativeMineResponse)
def mine_from_logs_endpoint(
    req: HardNegativeMineRequest = HardNegativeMineRequest(),
    db: Session = Depends(get_db),
):
    """Mines ambiguous inferences and potential false positives from recognition logs into review queue."""
    mined = mine_hard_negatives_from_logs(
        db,
        min_confidence=req.min_confidence,
        min_similarity=req.min_similarity,
        max_similarity=req.max_similarity,
        include_unknowns_only=req.include_unknowns_only,
        limit=req.limit,
    )
    return HardNegativeMineResponse(
        mined_count=len(mined),
        message=f"Successfully mined {len(mined)} candidate hard negatives from recent inference logs.",
        records=[format_hard_negative_response(r) for r in mined],
    )


@router.post("/hard-negatives/promote", response_model=PromoteHardNegativesResponse)
def promote_hard_negatives_endpoint(
    req: PromoteHardNegativesRequest,
    db: Session = Depends(get_db),
):
    """
    Safely promotes reviewed hard-negatives to train/val dataset splits.
    Strictly forbids promoting into 'test' to prevent test-set data leakage.
    """
    try:
        res = promote_hard_negatives_to_dataset(
            db,
            record_ids=req.record_ids,
            target_split=req.target_split,
            as_sku_id=req.as_sku_id,
        )
        return PromoteHardNegativesResponse(
            promoted_count=res["promoted_count"],
            target_split=res["target_split"],
            created_image_ids=res["created_image_ids"],
            message=res["message"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# 2. ERROR ANALYSIS PIPELINE ENDPOINTS
# ---------------------------------------------------------------------------

@router.post("/error-analysis", response_model=ErrorAnalysisResponse)
def run_error_analysis_endpoint(
    model_id: Optional[str] = Query(None, description="Model ID to evaluate (defaults to active)"),
    split: str = Query("val", description="Dataset split to evaluate ('val' or 'test')"),
    similarity_threshold: float = Query(0.60, ge=0.1, le=0.99),
    margin_threshold: float = Query(0.05, ge=0.0, le=0.5),
    db: Session = Depends(get_db),
):
    """
    Executes structured Error Analysis:
    - 10 recognition failure categories
    - Confusion Matrix
    - Per-SKU precision, recall, F1, accuracy
    - Separation of detector localization from embedding similarity
    - Representative error examples
    """
    try:
        return run_error_analysis(
            db,
            model_id=model_id,
            split=split,
            similarity_threshold=similarity_threshold,
            margin_threshold=margin_threshold,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error analysis failed: {str(e)}")


# ---------------------------------------------------------------------------
# 3. THRESHOLD CALIBRATION ENDPOINTS
# ---------------------------------------------------------------------------

@router.post("/threshold-calibration", response_model=ThresholdCalibrationResponse)
def run_threshold_calibration_endpoint(
    req: ThresholdCalibrationRequest = ThresholdCalibrationRequest(),
    db: Session = Depends(get_db),
):
    """
    Executes 2D similarity and margin threshold calibration strictly using validation split.
    Prohibits calibrating against test set to prevent leakage.
    """
    try:
        return run_threshold_calibration(db, req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Threshold calibration failed: {str(e)}")


# ---------------------------------------------------------------------------
# 4. REGRESSION BENCHMARK & MODEL PROMOTION SAFEGUARDS
# ---------------------------------------------------------------------------

@router.post("/benchmark", response_model=ComparativeBenchmarkResponse)
def run_benchmark_endpoint(
    req: ComparativeBenchmarkRequest,
    db: Session = Depends(get_db),
):
    """
    Compares candidate model against baseline model using Section 4 Promotion Safeguards.
    Evaluates overall accuracy, unknown FAR, and per-SKU stability.
    """
    try:
        return run_comparative_benchmark(db, req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark execution failed: {str(e)}")


@router.post("/promote-candidate", response_model=PromoteCandidateModelResponse)
def promote_candidate_endpoint(
    req: PromoteCandidateModelRequest,
    db: Session = Depends(get_db),
):
    """
    Safeguarded candidate model promotion endpoint.
    Verifies all Section 4 safeguards before atomically migrating gallery and activating candidate model.
    """
    try:
        return execute_model_promotion_safeguarded(
            db,
            candidate_model_id=req.candidate_model_id,
            force=req.force,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Promotion failed: {str(e)}")
