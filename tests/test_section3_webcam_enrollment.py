import io
import os
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.core.database import SessionLocal
from app.models.sku import SKU
from app.models.image import ProductImage
from app.models.embedding import Embedding
from app.models.model_registry import ModelRegistry
from app.services.model_registry_service import get_active_model

client = TestClient(app)


def create_test_image_bytes(color=(128, 64, 32), size=(200, 200), format="JPEG") -> bytes:
    import random
    img = Image.new("RGB", size, color=color)
    img.putpixel((0, 0), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    img.putpixel((1, 1), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


def test_recognition_status_endpoint():
    response = client.get("/api/v1/recognition/status")
    assert response.status_code == 200
    data = response.json()
    assert "active_detector" in data
    assert "active_embedding_model" in data
    assert "embedding_dim" in data
    assert data["embedding_dim"] in [960, 512]
    assert data["is_gallery_compatible"] is True
    assert data["gallery_reference_count"] >= 0


def test_few_shot_sku_enrollment_without_retraining():
    """
    SECTION 3: Test Few-Shot SKU Enrollment.
    Verifies that a new SKU can be registered with reference images,
    embeddings are indexed using the active model, and NO model training is triggered.
    """
    db = SessionLocal()
    try:
        active_model = get_active_model(db, "embedding")
        initial_model_id = active_model.model_id
        initial_dim = active_model.embedding_dim

        # Check existing training jobs count
        initial_jobs_res = client.get("/api/v1/training/jobs")
        initial_jobs_count = len(initial_jobs_res.json()) if initial_jobs_res.status_code == 200 else 0

        # Unique SKU code
        import uuid
        test_sku_code = f"FEWSHOT_{uuid.uuid4().hex[:6].upper()}"
        test_name = "Few-Shot Test Beverage"

        # Generate 2 reference images
        img1_bytes = create_test_image_bytes(color=(220, 50, 50))
        img2_bytes = create_test_image_bytes(color=(210, 60, 60))

        files = [
            ("reference_images", ("ref_angle1.jpg", img1_bytes, "image/jpeg")),
            ("reference_images", ("ref_angle2.jpg", img2_bytes, "image/jpeg")),
        ]
        data = {
            "sku_code": test_sku_code,
            "name": test_name,
            "category": "Beverages",
            "description": "Zero-shot enrolled product without retraining",
        }

        response = client.post("/api/v1/skus/enroll-few-shot", data=data, files=files)
        assert response.status_code == 201, response.text
        res_data = response.json()

        assert res_data["status"] == "success"
        assert res_data["sku_code"] == test_sku_code
        assert res_data["enrolled_images_count"] >= 1
        assert res_data["retrained"] is False
        assert res_data["model_id"] == initial_model_id
        assert res_data["embedding_dim"] == initial_dim

        # Verify database state
        sku_record = db.query(SKU).filter(SKU.sku_code == test_sku_code).first()
        assert sku_record is not None
        assert sku_record.is_active is True

        # Verify embeddings were persisted with correct active model and dimension
        embs = db.query(Embedding).filter(Embedding.sku_id == sku_record.id).all()
        assert len(embs) >= 1
        for emb in embs:
            assert emb.model_name == initial_model_id
            assert emb.embedding_dim == initial_dim
            assert len(emb.vector_bytes) == initial_dim * 4  # float32 = 4 bytes per dim

        # Verify NO new training job was launched
        post_jobs_res = client.get("/api/v1/training/jobs")
        post_jobs_count = len(post_jobs_res.json()) if post_jobs_res.status_code == 200 else 0
        assert post_jobs_count == initial_jobs_count

        # Test duplicate enrollment prevention
        dup_files = [
            ("reference_images", ("ref_angle1.jpg", img1_bytes, "image/jpeg")),
        ]
        dup_res = client.post("/api/v1/skus/enroll-few-shot", data=data, files=dup_files)
        assert dup_res.status_code == 409  # Conflict on duplicate sku_code

    finally:
        db.close()


def test_webcam_predict_frame_pipeline_and_unknown_handling():
    """
    SECTION 3: Test Webcam predict-frame endpoint.
    Verifies fast frame processing, margin check, and proper UNKNOWN rejection.
    """
    # Create a test frame
    frame_bytes = create_test_image_bytes(color=(40, 180, 70), size=(320, 240))

    files = {"file": ("webcam_frame.jpg", frame_bytes, "image/jpeg")}
    data = {
        "confidence_threshold": 0.50,
        "similarity_threshold": 0.85,  # High threshold to guarantee UNKNOWN on synthetic frame
        "margin_threshold": 0.05,
        "allow_fallback": "false",
        "frame_seq": "42",
    }

    response = client.post("/api/v1/recognition/predict-frame", data=data, files=files)
    assert response.status_code == 200
    res_data = response.json()

    assert "detected_objects" in res_data
    assert "detector_model" in res_data
    assert "embedding_model" in res_data
    assert "duration_ms" in res_data
    assert res_data["duration_ms"] > 0
    assert res_data["frame_seq"] == 42
    # With allow_fallback=False and no COCO objects detected in flat green image, detections count should be 0
    assert res_data["objects_detected_count"] == 0
    assert res_data["has_fallback_detection"] is False


def test_predict_frame_with_fallback_and_unknown_rejection():
    """
    When allow_fallback=true, flat image creates a fallback crop and evaluates embedding matching.
    High similarity threshold must reject it as UNKNOWN without forcing a known SKU.
    """
    frame_bytes = create_test_image_bytes(color=(100, 100, 100), size=(200, 200))

    files = {"file": ("query_test.jpg", frame_bytes, "image/jpeg")}
    data = {
        "confidence_threshold": 0.50,
        "similarity_threshold": 0.95,  # High threshold
        "margin_threshold": 0.05,
        "allow_fallback": "true",
    }

    response = client.post("/api/v1/recognition/predict", data=data, files=files)
    assert response.status_code == 200
    res_data = response.json()

    assert res_data["objects_detected_count"] >= 1
    obj = res_data["detected_objects"][0]
    # Because threshold is 0.95, synthetic grey crop must be rejected as UNKNOWN
    assert obj["is_unknown"] is True
    assert obj["recognition_status"] == "UNKNOWN"
    assert "UNKNOWN" in obj["predicted_sku"]
    assert obj["rejection_reason"] is not None
