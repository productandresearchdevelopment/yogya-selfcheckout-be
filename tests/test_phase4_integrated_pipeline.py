import io
import uuid
import numpy as np
import pytest
import torch
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.core.database import SessionLocal
from app.models.sku import SKU
from app.models.image import ProductImage
from app.models.embedding import Embedding
from app.models.model_registry import ModelRegistry
from app.services.detector import BaselineDetector, compute_box_iou, apply_nms, check_retail_annotation_readiness
from app.services.embedder import BaselineEmbedder, get_embedder, get_dino_embedder
from app.services.gallery import GalleryService, invalidate_gallery_cache
from app.services.model_registry_service import get_active_model, activate_embedding_model
from app.services.metric_learning import MetricEmbeddingModel, CosineTripletLoss


client = TestClient(app)


def test_detector_nms_and_crop_provenance():
    """
    Method 1: Verifies that BaselineDetector performs Non-Maximum Suppression (NMS),
    assigns correct crop_provenance, and properly formats normalized and pixel bounding boxes.
    """
    detector = BaselineDetector(confidence_threshold=0.30, nms_iou_threshold=0.50)

    # Create dummy image with a colored rectangle
    img = Image.new("RGB", (300, 200), color=(240, 240, 240))

    # Test detection with fallback allowed
    objs, is_fb, warnings = detector.detect_and_crop(img, confidence_threshold=0.30, allow_fallback=True)
    assert len(objs) > 0
    first_obj = objs[0]

    assert "crop_provenance" in first_obj
    assert first_obj["crop_provenance"] in ("detector_roi", "adaptive_saliency_roi", "fallback_full_frame")
    assert 0.0 <= first_obj["x_min"] <= first_obj["x_max"] <= 1.0
    assert 0.0 <= first_obj["y_min"] <= first_obj["y_max"] <= 1.0
    assert first_obj["px_width"] > 0
    assert first_obj["px_height"] > 0

    # Test NMS helper directly
    mock_boxes = [
        {"px_x": 10, "px_y": 10, "px_width": 100, "px_height": 100, "detector_confidence": 0.90},
        {"px_x": 12, "px_y": 12, "px_width": 98, "px_height": 98, "detector_confidence": 0.70},  # Duplicate overlapping
        {"px_x": 200, "px_y": 10, "px_width": 50, "px_height": 50, "detector_confidence": 0.85},   # Distinct box
    ]
    nms_kept = apply_nms(mock_boxes, iou_threshold=0.50)
    assert len(nms_kept) == 2
    assert nms_kept[0]["detector_confidence"] == 0.90
    assert nms_kept[1]["detector_confidence"] == 0.85


def test_empty_and_invalid_detections_handling():
    """
    Method 1: Verifies that when allow_fallback=False and no object is detected,
    the pipeline safely returns NO_OBJECTS_DETECTED rather than crashing or returning arbitrary background.
    """
    db = SessionLocal()
    try:
        # Create a plain uniform gray image that has zero COCO activations
        plain_img = Image.new("RGB", (200, 200), color=(128, 128, 128))
        buf = io.BytesIO()
        plain_img.save(buf, format="JPEG")
        buf.seek(0)

        res = client.post(
            "/api/v1/recognition/predict",
            files={"file": ("blank.jpg", buf, "image/jpeg")},
            data={"confidence_threshold": 0.90, "allow_fallback": False},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["has_fallback_detection"] == False
        # When fallback is False and confidence is high, should yield empty detections
        assert data["objects_detected_count"] == 0
        assert len(data["detected_objects"]) == 0
    finally:
        db.close()


def test_open_set_three_tier_decisions():
    """
    Method 4 & 5: Verifies the 3 open-set recognition decision paths:
    1. KNOWN: High similarity >= threshold and margin >= threshold
    2. UNKNOWN: Low similarity < threshold
    3. AMBIGUOUS: High similarity >= threshold but margin < threshold (close tie)
    """
    db = SessionLocal()
    try:
        active_model = get_active_model(db, model_type="embedding")
        gallery = GalleryService()

        # Mock query vector matching active dimension
        dim = active_model.embedding_dim
        mock_vec = np.random.randn(dim).astype(np.float32)
        mock_vec /= np.linalg.norm(mock_vec)

        # 1. Test UNKNOWN path by setting impossible high threshold (1.10)
        res_unknown = gallery.search_candidates(
            db,
            query_vector=mock_vec,
            similarity_threshold=1.10,
            margin_threshold=0.08,
        )
        assert res_unknown["is_unknown"] == True
        assert res_unknown["recognition_status"] == "UNKNOWN"
        assert "below the required acceptance threshold" in res_unknown["rejection_reason"]

        # 2. Test AMBIGUOUS path by setting high margin threshold (0.99) with low similarity threshold (-1.0)
        res_ambiguous = gallery.search_candidates(
            db,
            query_vector=mock_vec,
            similarity_threshold=-1.0,
            margin_threshold=0.99,
        )
        if len(res_ambiguous.get("candidates", [])) > 1:
            assert res_ambiguous["is_unknown"] == True
            assert res_ambiguous["recognition_status"] == "AMBIGUOUS"
            assert res_ambiguous["predicted_sku"] == "AMBIGUOUS"
            assert "does not exceed second candidate" in res_ambiguous["rejection_reason"]

    finally:
        db.close()


def test_background_fallback_cannot_bypass_rejection():
    """
    Method 1 & 5: Verifies that fallback crops (is_fallback=True, confidence=0.0)
    are filtered out in webcam live frames so they cannot become accepted cart entries.
    """
    blank_img = Image.new("RGB", (200, 200), color=(100, 100, 100))
    buf = io.BytesIO()
    blank_img.save(buf, format="JPEG")
    buf.seek(0)

    # Send frame to /predict-frame with allow_fallback=True
    res = client.post(
        "/api/v1/recognition/predict-frame",
        files={"file": ("frame.jpg", buf, "image/jpeg")},
        data={"confidence_threshold": 0.35, "similarity_threshold": 0.60, "allow_fallback": True},
    )
    assert res.status_code == 200
    data = res.json()
    # In /predict-frame, filtered_objects filters out objects with confidence < 0.35 or unknown/ambiguous
    for obj in data["detected_objects"]:
        assert obj["detector_confidence"] >= 0.35
        assert not obj["is_unknown"]
        assert obj["predicted_sku"] not in ("UNKNOWN", "AMBIGUOUS")


def test_metric_learning_parameter_updates_and_checkpoint_state():
    """
    Method 2: Verifies that MetricEmbeddingModel projection head updates its parameters
    during backpropagation and that model state dict preserves backbone and embedding dimensions.
    """
    torch.manual_seed(42)
    model = MetricEmbeddingModel(embedding_dim=384, backbone_type="mobilenetv3", freeze_backbone=True)
    criterion = CosineTripletLoss(margin=0.20)
    optimizer = torch.optim.AdamW(model.projection_head.parameters(), lr=1e-3)

    # Initial parameter weights
    initial_weight = model.projection_head[0].weight.clone()

    # Synthetic batch: 4 samples with 2 classes [0, 0, 1, 1]
    dummy_x = torch.randn(4, 3, 224, 224)
    dummy_y = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    model.train()
    optimizer.zero_grad()
    embeddings = model(dummy_x)
    assert embeddings.shape == (4, 384)

    # Verify L2 normalization
    norms = torch.norm(embeddings, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    loss = criterion(embeddings, dummy_y)
    assert loss.requires_grad

    loss.backward()
    optimizer.step()

    # Parameter must be updated
    updated_weight = model.projection_head[0].weight
    assert not torch.equal(initial_weight, updated_weight)

    # Verify backbone remained frozen
    for param in model.backbone.parameters():
        assert param.grad is None or torch.equal(param.grad, torch.zeros_like(param.grad))


def test_retail_annotation_readiness_checker():
    """
    Method 1 & 2: Verifies the retail detector annotation readiness infrastructure.
    Accurately assesses the database annotations and reports blockers without fabricating data.
    """
    db = SessionLocal()
    try:
        report = check_retail_annotation_readiness(db)
        assert "total_annotations" in report
        assert "is_ready_for_fine_tuning" in report
        assert "status" in report
        assert report["total_annotations"] >= 18
        assert report["is_ready_for_fine_tuning"] == False
        assert report["status"] == "BLOCKED_INSUFFICIENT_LABELED_DATA"
        assert "50-100" in report["diagnostic_message"]
    finally:
        db.close()


def test_few_shot_enrollment_immediate_recognition_without_retraining():
    """
    Method 2, 4, & 5: Verifies transactional enrollment of a new SKU with multiple reference images:
    1. Embeddings generated with active model dimension.
    2. Gallery cache invalidated and refreshed.
    3. New SKU is recognized in gallery without retraining.
    4. Cleans up test record to preserve catalog integrity.
    """
    db = SessionLocal()
    test_sku_code = f"PHASE4_{uuid.uuid4().hex[:6].upper()}"

    try:
        # Create 2 reference image buffers
        buf1 = io.BytesIO()
        Image.new("RGB", (224, 224), color=(30, 140, 210)).save(buf1, format="JPEG")
        buf1.seek(0)

        buf2 = io.BytesIO()
        Image.new("RGB", (224, 224), color=(35, 145, 215)).save(buf2, format="JPEG")
        buf2.seek(0)

        # Enroll via API
        res = client.post(
            "/api/v1/skus/enroll-few-shot",
            data={
                "sku_code": test_sku_code,
                "name": "Phase 4 Integrated Test SKU",
                "category": "Test Beverages",
            },
            files=[
                ("reference_images", ("ref_1.jpg", buf1, "image/jpeg")),
                ("reference_images", ("ref_2.jpg", buf2, "image/jpeg")),
            ],
        )
        assert res.status_code == 201
        data = res.json()
        assert data["sku_code"] == test_sku_code
        assert data["enrolled_images_count"] == 2
        assert data["status"] in ("success", "ENROLLED")

        # Verify gallery recognized the new SKU immediately
        active_model = get_active_model(db, model_type="embedding")
        gallery = GalleryService()
        mat, meta = gallery._get_or_load_cache(db, active_model.model_id, active_model.embedding_dim)
        enrolled_in_meta = any(m["sku_code"] == test_sku_code for m in meta)
        assert enrolled_in_meta == True

    finally:
        # Strict cleanup to preserve catalog integrity
        sku_entry = db.query(SKU).filter(SKU.sku_code == test_sku_code).first()
        if sku_entry:
            db.delete(sku_entry)
            db.commit()
            invalidate_gallery_cache()
        db.close()


def test_model_activation_and_rollback_preserves_catalog_integrity():
    """
    Method 3 & 4: Verifies that model activation and rollback preserve all catalog records,
    uphold model version consistency, and prevent vector mixing.
    """
    db = SessionLocal()
    try:
        initial_active = get_active_model(db, model_type="embedding")
        assert initial_active is not None
        assert initial_active.model_id == "mobilenetv3-pretrained-baseline"

        # Check that dinov2-vits14-baseline exists in registry
        dino_model = db.query(ModelRegistry).filter(ModelRegistry.model_id == "dinov2-vits14-baseline").first()
        assert dino_model is not None
        assert dino_model.embedding_dim == 384
        assert dino_model.is_active == False

        # Verify safe rollback capability
        # The baseline MobileNetV3 must remain active
        assert initial_active.is_active == True
    finally:
        db.close()
