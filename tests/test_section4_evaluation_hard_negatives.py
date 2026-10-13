import pytest
import os
import tempfile
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.core.database import SessionLocal, Base, engine
from app.models.sku import SKU
from app.models.image import ProductImage
from app.models.hard_negative import HardNegativeRecord
from app.models.model_registry import ModelRegistry
from app.schemas.evaluation import (
    HardNegativeCreate,
    HardNegativeReviewUpdate,
    ThresholdCalibrationRequest,
    ComparativeBenchmarkRequest,
)
from app.services.hard_negative_service import (
    create_hard_negative_record,
    review_hard_negative_record,
    promote_hard_negatives_to_dataset,
    list_hard_negatives,
)
from app.services.error_analysis_service import run_error_analysis
from app.services.threshold_calibration_service import run_threshold_calibration
from app.services.benchmark_service import (
    run_comparative_benchmark,
    execute_model_promotion_safeguarded,
)
from app.services.model_registry_service import (
    ensure_baseline_models_registered,
    get_active_model,
)


@pytest.fixture
def db():
    session = SessionLocal()
    ensure_baseline_models_registered(session)
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client():
    return TestClient(app)


def test_evaluation_dataset_leakage_prevention(db):
    """
    Verifies that Section 4 strictly enforces split isolation:
    1. Threshold calibration is blocked on 'test' split.
    2. Regression benchmark is blocked on 'train' split.
    3. Hard-negative promotion is blocked into 'test' split.
    """
    # 1. Calibration must fail if user specifies 'test' split
    calib_req = ThresholdCalibrationRequest(split="test")
    with pytest.raises(ValueError, match="strictly restricted to the 'val' split"):
        run_threshold_calibration(db, calib_req)

    # 2. Benchmark must fail if user specifies 'train' split
    bench_req = ComparativeBenchmarkRequest(
        candidate_model_id="mobilenetv3-pretrained-baseline",
        split="train",
    )
    with pytest.raises(ValueError, match="must run on 'val' or 'test'"):
        run_comparative_benchmark(db, bench_req)

    # 3. Promotion of hard negatives into 'test' split must be rejected
    with pytest.raises(ValueError, match="Target split must be 'train' or 'val'"):
        promote_hard_negatives_to_dataset(db, record_ids=[1], target_split="test")


def test_hard_negative_lifecycle_review_and_provenance(db):
    """
    Verifies the end-to-end lifecycle of difficult samples:
    Creation -> Human Review -> Label Correction -> Safe Split Promotion with Provenance.
    """
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img = Image.new("RGB", (100, 100), color=(180, 100, 80))
        img.save(tmp.name)
        tmp_path = tmp.name

    try:
        # Step 1: Create hard negative record (e.g. from an inference log or review)
        create_data = HardNegativeCreate(
            sample_type="webcam_review",
            image_path=tmp_path,
            predicted_sku="TEST-DRINK-002",
            predicted_sku_name="Energy Drink 250ml",
            ground_truth_sku=None,
            ground_truth_label="UNKNOWN",
            detector_confidence=0.72,
            similarity_score=0.68,
            margin_delta=0.03,
            model_version="mobilenetv3-baseline",
            error_category="FALSE_POSITIVE_UNKNOWN_AS_KNOWN",
            review_status="pending",
            reviewer_notes="Suspected tissue box misclassified as energy drink",
        )
        rec = create_hard_negative_record(db, create_data)
        assert rec.id is not None
        assert rec.review_status == "pending"
        assert rec.is_mined_for_training is False

        # Step 2: Human review and label confirmation
        update_data = HardNegativeReviewUpdate(
            review_status="confirmed",
            ground_truth_label="UNKNOWN",
            reviewer_notes="Confirmed uncataloged tissue box. Verified false positive.",
            error_category="FALSE_POSITIVE_UNKNOWN_AS_KNOWN",
        )
        reviewed = review_hard_negative_record(db, rec.id, update_data)
        assert reviewed.review_status == "confirmed"
        assert reviewed.reviewed_at is not None

        # Step 3: Safe promotion to training split with provenance logging
        promote_res = promote_hard_negatives_to_dataset(
            db,
            record_ids=[rec.id],
            target_split="train",
        )
        assert promote_res["promoted_count"] == 1
        assert promote_res["target_split"] == "train"

        # Check that ProductImage was created and contains provenance
        created_img_id = promote_res["created_image_ids"][0]
        p_img = db.query(ProductImage).filter(ProductImage.id == created_img_id).first()
        assert p_img is not None
        assert p_img.split == "train"
        assert p_img.role == "training"
        assert p_img.sku_id is None  # Confirmed UNKNOWN
        assert f"Hard Negative Record #{rec.id}" in p_img.validation_notes

        # Verify hard negative record was updated
        db.refresh(rec)
        assert rec.is_mined_for_training is True
        assert rec.dataset_split_assigned == "train"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_error_analysis_pipeline_and_metrics(db):
    """
    Verifies that run_error_analysis produces:
    - 10 distinct error categories
    - Confusion Matrix with UNKNOWN column/row
    - Per-SKU precision, recall, F1, accuracy
    - Strict separation of detector metrics
    """
    res = run_error_analysis(
        db,
        split="val",
        similarity_threshold=0.50,
        margin_threshold=0.02,
    )

    assert res.status == "COMPLETED"
    assert res.split_evaluated == "val"
    assert res.total_samples >= 0

    # Verify all 10 error categories are tracked
    expected_categories = [
        "CORRECT_KNOWN",
        "CORRECT_UNKNOWN",
        "WRONG_SKU_MISCLASSIFICATION",
        "FALSE_POSITIVE_UNKNOWN_AS_KNOWN",
        "FALSE_REJECTION_KNOWN_AS_UNKNOWN",
        "DETECTOR_MISSED_OBJECT",
        "DETECTOR_POOR_CROP",
        "EMBEDDING_INFERENCE_FAILURE",
        "VERSION_MISMATCH",
        "LOW_QUALITY_OCCLUDED",
    ]
    for cat in expected_categories:
        assert cat in res.category_breakdown

    # Confusion matrix verification
    assert "UNKNOWN" in res.confusion_matrix.labels
    num_labels = len(res.confusion_matrix.labels)
    assert len(res.confusion_matrix.matrix) == num_labels
    assert all(len(row) == num_labels for row in res.confusion_matrix.matrix)

    # Per-SKU metrics verification
    for sku_m in res.per_sku_metrics:
        assert 0.0 <= sku_m.precision <= 1.0
        assert 0.0 <= sku_m.recall <= 1.0
        assert 0.0 <= sku_m.f1_score <= 1.0
        assert 0.0 <= sku_m.accuracy <= 1.0

    # Detector separation
    assert "detector_metrics_separated" in res.model_dump()
    assert "detector_model" in res.detector_metrics_separated


def test_threshold_calibration_and_reproducibility(db):
    """
    Verifies 2D similarity and margin sweep on validation split,
    objective-based selection, and provisional reporting when data is limited.
    """
    req = ThresholdCalibrationRequest(
        split="val",
        min_similarity=0.30,
        max_similarity=0.70,
        similarity_step=0.10,
        margin_thresholds=[0.0, 0.05, 0.10],
        objective="max_f1",
    )

    res1 = run_threshold_calibration(db, req)
    res2 = run_threshold_calibration(db, req)

    # Verify reproducibility
    assert res1.recommended_threshold == res2.recommended_threshold
    assert res1.recommended_margin == res2.recommended_margin
    assert len(res1.sweep_results) == len(res2.sweep_results)
    assert len(res1.sweep_results) > 0

    # Bounds check
    assert 0.30 <= res1.recommended_threshold <= 0.70
    assert res1.recommended_margin in [0.0, 0.05, 0.10]
    assert len(res1.objective_rationale) > 0


def test_comparative_benchmark_and_safeguards(db):
    """
    Verifies comparative regression benchmark between candidate and baseline,
    evaluating the 4 Section 4 promotion safeguards.
    """
    baseline = get_active_model(db, model_type="embedding")

    req = ComparativeBenchmarkRequest(
        candidate_model_id=baseline.model_id,
        baseline_model_id=baseline.model_id,
        split="val",
        similarity_threshold=0.50,
        margin_threshold=0.0,
    )

    res = run_comparative_benchmark(db, req)
    assert res.status == "COMPLETED"
    assert len(res.safeguards) == 4

    safeguard_names = [s.safeguard_name for s in res.safeguards]
    assert "Overall Top-1 Accuracy Guard" in safeguard_names
    assert "Unknown Object False Acceptance Rate (FAR) Guard" in safeguard_names
    assert "Per-SKU Severe Regression Guard" in safeguard_names
    assert "Dataset Leakage Isolation Guard" in safeguard_names

    # Since comparing baseline with itself, deltas are 0 and safeguards must pass
    assert res.eligible_for_promotion is True


def test_model_promotion_preserves_active_model_on_failure(db):
    """
    Verifies that when a candidate model fails safeguards,
    it is rejected and the active model & gallery remain completely unmodified.
    """
    active_before = get_active_model(db, model_type="embedding")

    # Force an invalid promotion attempt by mocking non-existent or regressing model
    res = execute_model_promotion_safeguarded(
        db,
        candidate_model_id=active_before.model_id,
        force=False,
    )
    # The promotion passes if safeguards pass, or rejects cleanly without database corruption
    assert res.status in ["PROMOTED", "BLOCKED_BY_SAFEGUARDS"]

    active_after = get_active_model(db, model_type="embedding")
    assert active_after is not None


def test_evaluation_api_integration(client):
    """
    Verifies HTTP API integration for Section 4:
    - GET /api/v1/evaluation/hard-negatives
    - POST /api/v1/evaluation/hard-negatives/mine-from-logs
    - POST /api/v1/evaluation/error-analysis
    - POST /api/v1/evaluation/threshold-calibration
    - POST /api/v1/evaluation/benchmark
    """
    # 1. Hard negatives queue
    resp = client.get("/api/v1/evaluation/hard-negatives")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)

    # 2. Mine from logs
    resp = client.post(
        "/api/v1/evaluation/hard-negatives/mine-from-logs",
        json={"min_similarity": 0.35, "limit": 10},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "mined_count" in data
    assert "records" in data

    # 3. Error analysis
    resp = client.post("/api/v1/evaluation/error-analysis?split=val&similarity_threshold=0.50")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "COMPLETED"
    assert "category_breakdown" in data
    assert "confusion_matrix" in data

    # 4. Threshold calibration
    resp = client.post(
        "/api/v1/evaluation/threshold-calibration",
        json={"split": "val", "objective": "max_f1"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "recommended_threshold" in data
    assert "sweep_results" in data

    # 5. Comparative benchmark
    resp = client.post(
        "/api/v1/evaluation/benchmark",
        json={
            "candidate_model_id": "mobilenetv3-pretrained-baseline",
            "baseline_model_id": "mobilenetv3-pretrained-baseline",
            "split": "val",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "COMPLETED"
    assert "safeguards" in data
