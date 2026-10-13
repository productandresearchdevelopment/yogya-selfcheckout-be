import io
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.main import app
from app.services.embedder import BaselineEmbedder
from app.services.detector import BaselineDetector
from app.services.gallery import GalleryService
from app.core.database import SessionLocal
from app.models.sku import SKU
from app.models.image import ProductImage

client = TestClient(app)


def create_sample_image(text="SAMPLE", bg=(200, 200, 200), fg=(50, 50, 200)):
    img = Image.new("RGB", (320, 320), color=bg)
    draw = ImageDraw.Draw(img)
    draw.rectangle([60, 60, 260, 260], fill=fg, outline=(0, 0, 0), width=3)
    draw.text((70, 70), text, fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf, img


def test_embedder_l2_normalization():
    embedder = BaselineEmbedder()
    _, pil_img = create_sample_image("CHIPS", (220, 180, 50), (180, 50, 50))
    vec = embedder.extract_embedding_from_crop(pil_img)
    
    # Vector length must be 960 (MobileNetV3 feature dimension)
    assert len(vec) == 960
    # L2 norm must be approximately 1.0
    norm = np.linalg.norm(vec)
    assert np.isclose(norm, 1.0, atol=1e-4)


def test_detector_fallback_behavior():
    detector = BaselineDetector()
    # Create plain white image with no clear COCO object
    plain_img = Image.new("RGB", (300, 300), color=(255, 255, 255))
    buf = io.BytesIO()
    plain_img.save("temp_plain.jpg")

    try:
        # Detect with very high threshold where COCO will find no objects
        detections, is_fallback, warnings = detector.detect_and_crop("temp_plain.jpg", confidence_threshold=0.99)
        assert is_fallback is True
        assert len(detections) == 1
        assert detections[0]["is_fallback"] is True
        assert detections[0]["detector_confidence"] == 0.0
        assert detections[0]["x_min"] == 0.0
        assert detections[0]["x_max"] == 1.0
        assert len(warnings) > 0
    finally:
        import os
        if os.path.exists("temp_plain.jpg"):
            os.remove("temp_plain.jpg")


import uuid

def test_recognition_predict_pipeline_and_unknown_rejection():
    # 1. Register SKU
    sku_code = f"TEST-CHOCO-{uuid.uuid4().hex[:6].upper()}"
    client.post("/api/v1/skus", json={
        "sku_code": sku_code,
        "name": "Dark Chocolate 70% 100g",
    })
    sku_res = client.get("/api/v1/skus")
    target_sku = next(s for s in sku_res.json() if s["sku_code"] == sku_code)
    sku_id = target_sku["id"]

    # 2. Upload reference image
    buf1, _ = create_sample_image(f"CHOC_{uuid.uuid4().hex[:6]}", (60, 30, 20), (150, 75, 0))
    upload_res = client.post(
        "/api/v1/datasets/upload",
        data={"sku_id": sku_id, "role": "reference", "split": "reference"},
        files={"files": ("choco_ref.jpg", buf1.getvalue(), "image/jpeg")},
    )
    assert upload_res.status_code == 201

    # 3. Predict with matching/similar query image and normal threshold
    buf_query, _ = create_sample_image("CHOCOLATE", (65, 35, 25), (155, 80, 5))
    pred_res = client.post(
        "/api/v1/recognition/predict",
        data={"confidence_threshold": 0.20, "similarity_threshold": 0.40, "top_k": 3},
        files={"file": ("query_test.jpg", buf_query.getvalue(), "image/jpeg")},
    )
    assert pred_res.status_code == 200
    res_data = pred_res.json()
    assert res_data["objects_detected_count"] >= 1
    assert "duration_ms" in res_data
    assert "detector_model" in res_data
    assert "embedding_model" in res_data

    # Verify detection confidence and embedding similarity are separate fields
    first_obj = res_data["detected_objects"][0]
    assert "detector_confidence" in first_obj
    assert "top_similarity_score" in first_obj
    assert "px_x" in first_obj
    assert "px_width" in first_obj
    assert len(first_obj["candidates"]) > 0

    # 4. Predict with an impossibly high similarity threshold (e.g. 0.999) to verify UNKNOWN rejection rule
    pred_unknown_res = client.post(
        "/api/v1/recognition/predict",
        data={"confidence_threshold": 0.20, "similarity_threshold": 0.9999, "top_k": 3},
        files={"file": ("query_test2.jpg", buf_query.getvalue(), "image/jpeg")},
    )
    assert pred_unknown_res.status_code == 200
    unknown_data = pred_unknown_res.json()
    assert unknown_data["detected_objects"][0]["is_unknown"] is True
    assert unknown_data["detected_objects"][0]["predicted_sku"] == "UNKNOWN"
    assert "below the required acceptance threshold" in unknown_data["detected_objects"][0]["rejection_reason"]


def test_system_health_and_dashboard_stats():
    health_res = client.get("/api/v1/system/health")
    assert health_res.status_code == 200
    h_data = health_res.json()
    assert h_data["status"] == "HEALTHY"
    assert h_data["torch_available"] is True
    assert "storage_paths" in h_data

    dash_res = client.get("/api/v1/system/dashboard")
    assert dash_res.status_code == 200
    d_data = dash_res.json()
    assert d_data["sku_count"] >= 1
    assert d_data["image_count"] >= 1
    assert "recent_runs" in d_data
