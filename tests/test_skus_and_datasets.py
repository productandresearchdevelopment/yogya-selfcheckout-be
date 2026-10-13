import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.main import app
from app.core.database import Base, engine, SessionLocal
from app.models.sku import SKU
from app.models.image import ProductImage

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield
    # Cleanup will be done as needed


def create_test_image_bytes(color=(255, 0, 0), size=(300, 300), pattern=True, seed_text=None) -> io.BytesIO:
    img = Image.new("RGB", size, color=color)
    if pattern:
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 200, 200], fill=(0, 255, 0), outline=(0, 0, 255), width=4)
        txt = seed_text or f"PROD_{uuid.uuid4().hex[:6]}"
        draw.text((60, 60), txt, fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


import uuid

def test_sku_crud_and_duplicate_prevention():
    sku_code = f"TEST-SNACK-{uuid.uuid4().hex[:6].upper()}"
    
    # 1. Register new SKU
    res = client.post("/api/v1/skus", json={
        "sku_code": sku_code,
        "name": "Test Snack Crisps 50g",
        "category": "Snacks",
        "description": "Crispy salted snack",
    })
    assert res.status_code == 201
    data = res.json()
    assert data["sku_code"] == sku_code
    assert data["is_active"] is True
    sku_id = data["id"]

    # 2. Prevent duplicate SKU code
    dup_res = client.post("/api/v1/skus", json={
        "sku_code": sku_code,
        "name": "Duplicate Snack",
    })
    assert dup_res.status_code == 409
    assert "already exists" in dup_res.json()["detail"]

    # 3. Retrieve SKU
    get_res = client.get(f"/api/v1/skus/{sku_id}")
    assert get_res.status_code == 200
    assert get_res.json()["name"] == "Test Snack Crisps 50g"

    # 4. Update SKU
    patch_res = client.patch(f"/api/v1/skus/{sku_id}", json={
        "name": "Updated Test Snack 55g",
    })
    assert patch_res.status_code == 200
    assert patch_res.json()["name"] == "Updated Test Snack 55g"

    # 5. Deactivate SKU
    deact_res = client.post(f"/api/v1/skus/{sku_id}/deactivate")
    assert deact_res.status_code == 200
    assert deact_res.json()["is_active"] is False


def test_dataset_upload_quality_and_duplicates():
    # Create SKU for testing
    sku_code = f"TEST-DRINK-{uuid.uuid4().hex[:6].upper()}"
    sku_res = client.post("/api/v1/skus", json={
        "sku_code": sku_code,
        "name": "Energy Drink 250ml",
    })
    assert sku_res.status_code == 201
    sku_id = sku_res.json()["id"]

    seed_token = f"TEST_SEED_{uuid.uuid4().hex}"
    img_buf1 = create_test_image_bytes(color=(100, 150, 200), pattern=True, seed_text=seed_token)
    
    # 1. Upload reference image
    upload_res = client.post(
        "/api/v1/datasets/upload",
        data={"sku_id": sku_id, "role": "reference", "split": "reference"},
        files={"files": ("energy_drink_ref.jpg", img_buf1.getvalue(), "image/jpeg")},
    )
    assert upload_res.status_code == 201
    uploaded_data = upload_res.json()
    assert len(uploaded_data) == 1
    image_id = uploaded_data[0]["id"]
    assert uploaded_data[0]["role"] == "reference"
    assert uploaded_data[0]["width"] == 300
    assert uploaded_data[0]["height"] == 300
    assert uploaded_data[0]["blur_score"] > 0
    assert uploaded_data[0]["has_embedding"] is True

    # 2. Prevent duplicate file upload (same bytes = same SHA-256 hash)
    img_buf2 = create_test_image_bytes(color=(100, 150, 200), pattern=True, seed_text=seed_token)
    dup_res = client.post(
        "/api/v1/datasets/upload",
        data={"sku_id": sku_id, "role": "training", "split": "train"},
        files={"files": ("energy_drink_copy.jpg", img_buf2.getvalue(), "image/jpeg")},
    )
    assert dup_res.status_code == 409
    assert "Duplicate file detected" in dup_res.json()["detail"]

    # 3. Add bounding box annotation to image
    ann_res = client.post(
        f"/api/v1/datasets/images/{image_id}/annotations",
        json={
            "x_min": 0.15,
            "y_min": 0.15,
            "x_max": 0.70,
            "y_max": 0.70,
            "label": "TEST-DRINK-002",
            "confidence": 1.0,
            "source": "manual",
        },
    )
    assert ann_res.status_code == 201
    ann_data = ann_res.json()
    assert ann_data["px_x"] > 0
    assert ann_data["label"] == "TEST-DRINK-002"

    # 4. Export YOLO dataset
    yolo_res = client.post("/api/v1/datasets/export/yolo")
    assert yolo_res.status_code == 200
    assert yolo_res.json()["format"] == "YOLO"
    assert yolo_res.json()["total_images"] >= 1

    # 5. Export COCO dataset
    coco_res = client.post("/api/v1/datasets/export/coco")
    assert coco_res.status_code == 200
    assert coco_res.json()["format"] == "COCO"
