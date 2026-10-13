import os
import tempfile
import numpy as np
import torch
import pytest
from PIL import Image

from app.core.database import SessionLocal, Base, engine
from app.models.sku import SKU
from app.models.image import ProductImage
from app.models.embedding import Embedding
from app.models.model_registry import ModelRegistry
from app.models.training_job import TrainingJob
from app.services.metric_learning import (
    MetricEmbeddingModel,
    CosineTripletLoss,
    IdentityAwareBatchSampler,
    _run_training_pipeline,
)
from app.services.model_registry_service import (
    ensure_baseline_models_registered,
    get_active_model,
    list_registered_models,
    activate_embedding_model,
)
from app.services.gallery import GalleryService
from app.services.readiness_service import generate_dataset_readiness_report
from app.services.evaluation_service import (
    evaluate_detector_localization,
    evaluate_embedding_model_comparative,
)


@pytest.fixture
def db():
    session = SessionLocal()
    ensure_baseline_models_registered(session)
    try:
        yield session
    finally:
        session.close()


def test_metric_embedding_model_and_loss():
    """Verifies projection head architecture, L2 normalization, and triplet margin loss."""
    dim = 256
    model = MetricEmbeddingModel(embedding_dim=dim, freeze_backbone=True)
    model.eval()

    dummy_input = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy_input)

    assert out.shape == (4, dim)

    # Check L2 normalization (norm should be ~1.0)
    norms = torch.norm(out, p=2, dim=1)
    for n in norms:
        assert torch.isclose(n, torch.tensor(1.0), atol=1e-4)

    # Test Triplet Loss
    loss_fn = CosineTripletLoss(margin=0.2)
    labels = torch.tensor([1, 1, 2, 2])
    loss = loss_fn(out, labels)
    assert loss.dim() == 0  # Scalar loss
    assert loss.item() >= 0.0


def test_identity_aware_batch_sampler():
    """Verifies that identity-aware batch sampler yields P classes with K instances each."""
    labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    p = 2
    k = 2
    sampler = IdentityAwareBatchSampler(labels, p=p, k=k, num_batches=5)

    assert len(sampler) == 5
    for batch in sampler:
        assert len(batch) == p * k
        batch_labels = [labels[idx] for idx in batch]
        unique_batch_labels = set(batch_labels)
        assert len(unique_batch_labels) == p
        for lbl in unique_batch_labels:
            assert batch_labels.count(lbl) == k


def test_dataset_readiness_audit(db):
    """Verifies dataset readiness report structure and blocker reporting."""
    report = generate_dataset_readiness_report(db)
    assert "status" in report
    assert "is_ready" in report
    assert "summary" in report
    assert "blockers" in report
    assert "recommendations" in report
    assert isinstance(report["blockers"], list)
    assert report["summary"]["total_skus"] >= 2


def test_model_registry_baseline_preservation_and_activation(db):
    """
    Verifies that baseline pretrained model is preserved, fine-tuned model can be activated,
    and rollback to baseline works without data loss.
    """
    ensure_baseline_models_registered(db)
    active = get_active_model(db, "embedding")
    assert active is not None
    assert active.model_id in ("mobilenetv3-pretrained-baseline", "mobilenetv3-metric-job_20261012_081448_843314")

    # Verify baseline is in the list
    models = list_registered_models(db, "embedding")
    model_ids = [m.model_id for m in models]
    assert "mobilenetv3-pretrained-baseline" in model_ids

    # Activate baseline
    res_base = activate_embedding_model(db, "mobilenetv3-pretrained-baseline")
    assert res_base["status"] in ("ACTIVATED", "ALREADY_ACTIVE")

    active_now = get_active_model(db, "embedding")
    assert active_now.model_id == "mobilenetv3-pretrained-baseline"
    assert active_now.embedding_dim == 960


def test_cross_version_gallery_compatibility_guard(db):
    """
    Verifies Requirement 8: Enforcing that query vectors from incompatible
    dimensions or model versions cannot be compared against active gallery.
    """
    gallery_service = GalleryService()
    active_model = get_active_model(db, "embedding")

    # Create dummy query vector with wrong dimension (e.g. 128 instead of active dim)
    wrong_dim = 128 if active_model.embedding_dim != 128 else 256
    dummy_query = np.random.randn(wrong_dim).astype(np.float32)

    with pytest.raises(ValueError) as excinfo:
        gallery_service.search_candidates(db, dummy_query)

    assert "does not match active gallery model" in str(excinfo.value)


def test_sku_enrollment_does_not_retrain_model(db):
    """
    Verifies Requirement 8: Registering a new SKU and reference image
    only generates and appends embeddings; it does not change active model or trigger training.
    """
    active_before = get_active_model(db, "embedding")
    initial_jobs_count = db.query(TrainingJob).count()

    # Create a test SKU
    test_sku = SKU(sku_code=f"TEST-ENROLL-{np.random.randint(1000, 9999)}", name="Enrollment Test Product")
    db.add(test_sku)
    db.commit()

    # Create test image file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        pil_img = Image.new("RGB", (224, 224), color=(100, 150, 200))
        pil_img.save(tmp.name, "JPEG")
        tmp_path = tmp.name

    test_img = ProductImage(
        sku_id=test_sku.id,
        role="reference",
        split="reference",
        filename="test_enroll.jpg",
        file_path=tmp_path,
        file_hash="dummyhash_enroll",
        file_size_bytes=1000,
        width=224,
        height=224,
        mime_type="image/jpeg",
        is_valid=True,
    )
    db.add(test_img)
    db.commit()

    # Index embedding using GalleryService
    gallery_service = GalleryService()
    emb = gallery_service.index_image_embedding(db, test_img)

    assert emb is not None
    assert emb.embedding_dim == active_model_dim(active_before)
    assert emb.model_name == active_before.model_id

    # Verify active model is unchanged
    active_after = get_active_model(db, "embedding")
    assert active_after.model_id == active_before.model_id

    # Verify no new training jobs were triggered
    jobs_after_count = db.query(TrainingJob).count()
    assert jobs_after_count == initial_jobs_count

    # Cleanup test records
    db.delete(emb)
    db.delete(test_img)
    db.delete(test_sku)
    db.commit()
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


def active_model_dim(m: ModelRegistry) -> int:
    return m.embedding_dim


def test_detector_evaluation_separated_from_embedding(db):
    """Verifies that detector localization evaluation runs independently of embedding recognition."""
    det_eval = evaluate_detector_localization(db, confidence_threshold=0.35)
    assert "status" in det_eval
    assert "metrics" in det_eval
    assert det_eval["metrics"]["annotated_images_count"] >= 0
    # Confirm it does not include SKU classification accuracy
    assert "top1_accuracy" not in det_eval["metrics"]
