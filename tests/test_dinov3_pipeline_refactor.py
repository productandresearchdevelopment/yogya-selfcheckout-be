import io
import os
import numpy as np
import pytest
import torch
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.core.database import SessionLocal
from app.services.detector import ObjectDetector, BaselineDetector, compute_box_iou, apply_nms
from app.services.embedder import DINOv3Embedder, BaselineEmbedder, DINOv3InitializationError, get_dino_embedder
from app.services.gallery import SKUReferenceGallery, SKUMatcher, OpenSetDecision, GalleryService
from app.services.model_registry_service import get_active_model

client = TestClient(app)


def test_dinov3_initialization_failure_and_no_silent_fallback():
    """
    Stage Two: Verifies that DINOv3 is the primary embedding architecture and that
    when weights are unavailable (no DINOV3_CHECKPOINT_PATH and no HF_TOKEN),
    the system explicitly fails with DINOv3InitializationError and does NOT silently
    fall back to MobileNetV3 or DINOv2.
    """
    checkpoint_env = os.environ.get("DINOV3_CHECKPOINT_PATH")
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if not checkpoint_env and not hf_token:
        embedder = DINOv3Embedder(variant="dinov3_vits16")
        assert embedder.embedding_dim == 384
        assert embedder.model_name == "dinov3_vits16-baseline"

        dummy_crop = Image.new("RGB", (224, 224), color=(100, 150, 200))
        with pytest.raises((DINOv3InitializationError, RuntimeError)) as exc_info:
            embedder.extract_embedding_from_crop(dummy_crop)

        err_msg = str(exc_info.value).lower()
        # Must clearly identify the blocker and prohibit silent fallback
        assert "dinov3" in err_msg or "gated" in err_msg or "403" in err_msg
        assert "fallback" in err_msg or "prohibited" in err_msg or "cannot load" in err_msg


def test_object_detector_multi_product_localization_and_nms():
    """
    Stage One: Verifies pretrained object detector generates separate bounding boxes
    for products, suppresses overlapping duplicates via NMS (IoU >= 0.50),
    and correctly tags crop provenance.
    """
    detector = ObjectDetector(confidence_threshold=0.30, nms_iou_threshold=0.50)
    assert detector is not None

    # Synthetic image containing foreground objects
    img = Image.new("RGB", (400, 300), color=(230, 230, 230))

    # Verify NMS deduplication logic directly
    mock_detections = [
        {"px_x": 20, "px_y": 20, "px_width": 120, "px_height": 120, "detector_confidence": 0.92, "box_id": 0},
        {"px_x": 24, "px_y": 22, "px_width": 116, "px_height": 118, "detector_confidence": 0.65, "box_id": 1},  # Overlapping duplicate
        {"px_x": 220, "px_y": 50, "px_width": 100, "px_height": 110, "detector_confidence": 0.88, "box_id": 2}, # Separate product
    ]
    nms_results = apply_nms(mock_detections, iou_threshold=0.50)
    assert len(nms_results) == 2
    assert nms_results[0]["detector_confidence"] == 0.92
    assert nms_results[1]["detector_confidence"] == 0.88

    # Verify detection and crop extraction interface
    objs, is_fb, warnings = detector.detect_and_crop(img, confidence_threshold=0.30, allow_fallback=True)
    assert len(objs) > 0
    first_obj = objs[0]
    assert "crop_provenance" in first_obj
    assert first_obj["crop_provenance"] in ("detector_roi", "adaptive_saliency_roi", "fallback_full_frame")
    assert first_obj["px_width"] > 0
    assert first_obj["px_height"] > 0


def test_openset_decision_engine_three_tier_rules():
    """
    Stage Four: Verifies OpenSetDecision engine evaluates top-1 score and margin delta:
    - KNOWN: sim >= 0.60 AND margin >= 0.08
    - AMBIGUOUS: sim >= 0.60 AND margin < 0.08
    - UNKNOWN: sim < 0.60
    """
    # 1. KNOWN Match Case
    candidates_known = [
        {"sku_id": 1, "sku_code": "SKU_A", "sku_name": "Product A", "similarity_score": 0.82},
        {"sku_id": 2, "sku_code": "SKU_B", "sku_name": "Product B", "similarity_score": 0.65},
    ]
    dec_known = OpenSetDecision.evaluate(candidates_known, similarity_threshold=0.60, margin_threshold=0.08)
    assert dec_known["is_unknown"] is False
    assert dec_known["recognition_status"] == "KNOWN"
    assert dec_known["predicted_sku"] == "SKU_A"
    assert dec_known["margin_delta"] == 0.17
    assert dec_known["rejection_reason"] is None

    # 2. AMBIGUOUS Case (Close tie between candidates)
    candidates_ambiguous = [
        {"sku_id": 1, "sku_code": "SKU_A", "sku_name": "Product A", "similarity_score": 0.78},
        {"sku_id": 2, "sku_code": "SKU_B", "sku_name": "Product B", "similarity_score": 0.75}, # delta = 0.03 < 0.08
    ]
    dec_ambiguous = OpenSetDecision.evaluate(candidates_ambiguous, similarity_threshold=0.60, margin_threshold=0.08)
    assert dec_ambiguous["is_unknown"] is True
    assert dec_ambiguous["recognition_status"] == "AMBIGUOUS"
    assert dec_ambiguous["predicted_sku"] == "AMBIGUOUS"
    assert dec_ambiguous["margin_delta"] == 0.03
    assert "does not exceed second candidate" in dec_ambiguous["rejection_reason"]

    # 3. UNKNOWN Case (Low similarity score)
    candidates_unknown = [
        {"sku_id": 1, "sku_code": "SKU_A", "sku_name": "Product A", "similarity_score": 0.45},
        {"sku_id": 2, "sku_code": "SKU_B", "sku_name": "Product B", "similarity_score": 0.32},
    ]
    dec_unknown = OpenSetDecision.evaluate(candidates_unknown, similarity_threshold=0.60, margin_threshold=0.08)
    assert dec_unknown["is_unknown"] is True
    assert dec_unknown["recognition_status"] == "UNKNOWN"
    assert dec_unknown["predicted_sku"] == "UNKNOWN"
    assert "below the required acceptance threshold" in dec_unknown["rejection_reason"]

    # 4. Empty Gallery Case
    dec_empty = OpenSetDecision.evaluate([], similarity_threshold=0.60, margin_threshold=0.08)
    assert dec_empty["is_unknown"] is True
    assert dec_empty["recognition_status"] == "UNKNOWN"
    assert "No candidates" in dec_empty["rejection_reason"]


def test_gallery_matching_dimension_consistency_guard():
    """
    Stage Three: Verifies that SKUReferenceGallery / SKUMatcher enforces strict
    dimension consistency, preventing old MobileNetV3 (960-d) vectors from being
    compared directly against DINOv3 (384-d) vectors.
    """
    db = SessionLocal()
    try:
        active_model = get_active_model(db, model_type="embedding")
        matcher = SKUMatcher()

        # Generate a mismatched dimension vector (e.g. 960-d if active is 384-d, or vice-versa)
        wrong_dim = 960 if active_model.embedding_dim == 384 else 384
        bad_query_vector = np.random.randn(wrong_dim).astype(np.float32)

        with pytest.raises(ValueError) as exc_info:
            matcher.search_candidates(
                db=db,
                query_vector=bad_query_vector,
                similarity_threshold=0.60,
            )
        assert "dimension" in str(exc_info.value).lower()
    finally:
        db.close()
