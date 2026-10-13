import os
import time
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
from app.services.embedder import get_dino_embedder, BaselineEmbedder
from app.services.metric_learning import (
    MetricEmbeddingModel,
    CosineTripletLoss,
    _run_training_pipeline,
)
from app.services.model_registry_service import (
    ensure_baseline_models_registered,
    get_active_model,
    load_model_instance,
    activate_embedding_model,
)
from app.services.gallery import GalleryService, invalidate_gallery_cache
from app.services.detector import BaselineDetector
from app.core.config import DEFAULT_SIMILARITY_THRESHOLD, DEFAULT_MARGIN_THRESHOLD


@pytest.fixture
def db():
    session = SessionLocal()
    ensure_baseline_models_registered(session)
    try:
        yield session
    finally:
        session.close()


def test_dino_checkpoint_loading_and_embedding_dimension_consistency():
    """
    Rule 2, 4, 6: Verifies DINO backbone loading, embedding dimension consistency,
    and L2-normalized feature representation.
    """
    model, device, transform = get_dino_embedder("dinov2_vits14")
    assert model is not None
    assert device is not None
    assert transform is not None

    # Dummy RGB image crop (224, 224)
    dummy_img = Image.new("RGB", (224, 224), color=(120, 180, 240))
    tensor = transform(dummy_img).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model(tensor).squeeze().cpu().numpy().astype(np.float32)

    # DINO ViT-S/14 raw feature dimension is 384
    assert features.shape == (384,)
    norm = np.linalg.norm(features)
    assert norm > 1e-6

    # Test MetricEmbeddingModel with DINO backbone and projection head
    metric_model = MetricEmbeddingModel(
        embedding_dim=512,
        backbone_type="dinov2_vits14",
        freeze_backbone=True,
    )
    metric_model.to(device)
    metric_model.eval()

    # Backbone weights should be frozen
    for p in metric_model.backbone.parameters():
        assert p.requires_grad is False

    # Projection head weights must be trainable
    for p in metric_model.projection_head.parameters():
        assert p.requires_grad is True

    with torch.no_grad():
        projected = metric_model(tensor).squeeze().cpu().numpy().astype(np.float32)

    assert projected.shape == (512,)
    # Output must be L2 normalized on unit sphere
    proj_norm = np.linalg.norm(projected)
    assert abs(proj_norm - 1.0) < 1e-4


def test_dinov3_blocker_reporting_and_explicit_separation():
    """
    Rule 2 & 3: Checks whether official DINOv3 weights can be loaded in the current environment.
    Reports the blocker when gated access is unavailable without fabricating success.
    """
    checkpoint_env = os.environ.get("DINOV3_CHECKPOINT_PATH")
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if not checkpoint_env and not hf_token:
        # Must raise informative RuntimeError describing Meta license / gated access blocker
        with pytest.raises(RuntimeError) as exc_info:
            get_dino_embedder("dinov3_vits16")
        err_msg = str(exc_info.value)
        assert "Cannot load official" in err_msg or "gated" in err_msg.lower() or "dinov3" in err_msg.lower()

    # Meanwhile DINOv2 must be available as an explicitly separate, verified open baseline
    dino2_model, _, _ = get_dino_embedder("dinov2_vits14")
    assert dino2_model is not None


def test_dino_metric_learning_training_and_checkpoint_reload(db):
    """
    Rule 4 & 5: Verifies DINO-compatible metric-learning training loop,
    Identity-aware batching, checkpoint serialization, and reload.
    """
    # Create test SKUs and reference images
    sku1 = SKU(sku_code=f"DINO_TRN_A_{int(time.time()*1000)%10000}", name="DINO Train SKU A", is_active=True)
    sku2 = SKU(sku_code=f"DINO_TRN_B_{int(time.time()*1000)%10000}", name="DINO Train SKU B", is_active=True)
    db.add_all([sku1, sku2])
    db.commit()

    temp_files = []
    try:
        # Create 4 training images (2 per SKU for positive & negative pairs)
        for sku, color in [(sku1, (255, 0, 0)), (sku1, (240, 20, 20)), (sku2, (0, 0, 255)), (sku2, (20, 20, 240))]:
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            img = Image.new("RGB", (224, 224), color=color)
            img.save(tmp.name)
            tmp.close()
            temp_files.append(tmp.name)

            p_img = ProductImage(
                sku_id=sku.id,
                role="training",
                split="train",
                filename=os.path.basename(tmp.name),
                file_path=tmp.name,
                file_hash=f"hash_{os.path.basename(tmp.name)}",
                file_size_bytes=1024,
                width=224,
                height=224,
                mime_type="image/jpeg",
                is_valid=True,
            )
            db.add(p_img)
        db.commit()

        # Run 1 epoch training pipeline with DINO backbone
        job_id = f"test_dino_train_{int(time.time())}"
        config = {
            "backbone_type": "dinov2_vits14",
            "epochs": 1,
            "embedding_dim": 384,
            "learning_rate": 1e-4,
            "margin": 0.2,
            "p_classes": 2,
            "k_instances": 2,
            "freeze_backbone": True,
        }

        # Create training job entry
        from app.models.training_job import TrainingJob
        job = TrainingJob(
            job_id=job_id,
            model_id=f"dinov2_vits14-metric-{job_id}",
            status="QUEUED",
            total_epochs=1,
            current_epoch=0,
        )
        db.add(job)
        db.commit()

        _run_training_pipeline(job_id, config, SessionLocal)

        db.refresh(job)
        assert job.status == "COMPLETED"
        assert job.checkpoint_path is not None
        assert os.path.exists(job.checkpoint_path)

        # Verify checkpoint reload via model registry
        reg = db.query(ModelRegistry).filter(ModelRegistry.model_id == f"dinov2_vits14-metric-{job_id}").first()
        assert reg is not None
        assert reg.embedding_dim == 384

        loaded_model, dev, tf = load_model_instance(reg)
        assert loaded_model is not None
        # Forward pass test
        dummy_t = torch.randn(1, 3, 224, 224, device=dev)
        with torch.no_grad():
            out = loaded_model(dummy_t)
        assert out.shape == (1, 384)
        assert abs(torch.norm(out, p=2).item() - 1.0) < 1e-4

    finally:
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)


def test_dino_similar_sku_discrimination(db):
    """
    Rule 6: Evaluates discriminative power of DINO embeddings on fine-grained
    packaging differences compared to generic MobileNetV3 using real reference images.
    """
    dino_embedder = BaselineEmbedder(model_id="dinov2-vits14-baseline")
    mobilenet_embedder = BaselineEmbedder(model_id="mobilenetv3-pretrained-baseline")

    # Fetch reference images from 2 distinct SKUs in database
    skus = db.query(SKU).filter(SKU.is_active == True).all()
    valid_skus = [s for s in skus if s.images and os.path.exists(s.images[0].file_path)]

    if len(valid_skus) >= 2:
        img_a = Image.open(valid_skus[0].images[0].file_path)
        img_b = Image.open(valid_skus[1].images[0].file_path)
    else:
        arr_a = np.zeros((224, 224, 3), dtype=np.uint8)
        arr_a[:, ::16, 0] = 255
        img_a = Image.fromarray(arr_a)
        arr_b = np.zeros((224, 224, 3), dtype=np.uint8)
        arr_b[::16, :, 2] = 255
        img_b = Image.fromarray(arr_b)

    dino_vec_a = dino_embedder.extract_embedding_from_crop(img_a)
    dino_vec_b = dino_embedder.extract_embedding_from_crop(img_b)

    mb_vec_a = mobilenet_embedder.extract_embedding_from_crop(img_a)
    mb_vec_b = mobilenet_embedder.extract_embedding_from_crop(img_b)

    dino_sim = float(np.dot(dino_vec_a, dino_vec_b))
    mb_sim = float(np.dot(mb_vec_a, mb_vec_b))

    assert dino_sim < mb_sim
    assert dino_sim < 0.80
    assert abs(np.linalg.norm(dino_vec_a) - 1.0) < 1e-4
    assert abs(np.linalg.norm(dino_vec_b) - 1.0) < 1e-4


def test_open_set_recognition_and_top_two_margin_gate(db):
    """
    Rule 9: Verifies that the top-two margin gate is actually applied
    and rejects ambiguous candidate proposals as UNKNOWN.
    """
    gallery_service = GalleryService()

    sku1 = SKU(sku_code=f"MRG_SKU_1_{int(time.time()*1000)%10000}", name="Margin SKU 1", is_active=True)
    sku2 = SKU(sku_code=f"MRG_SKU_2_{int(time.time()*1000)%10000}", name="Margin SKU 2", is_active=True)
    db.add_all([sku1, sku2])
    db.commit()

    active_model = get_active_model(db, model_type="embedding")
    dim = active_model.embedding_dim

    # Base reference vector
    base_v = np.random.randn(dim).astype(np.float32)
    base_v = base_v / np.linalg.norm(base_v)

    # Orthogonal component to control exact cosine distance
    ortho_v = np.random.randn(dim).astype(np.float32)
    ortho_v = ortho_v - np.dot(ortho_v, base_v) * base_v
    ortho_v = ortho_v / np.linalg.norm(ortho_v)

    # Construct close_v with exact similarity 0.995 (delta 0.005 < margin_threshold 0.08)
    target_sim = 0.995
    close_v = (target_sim * base_v + np.sqrt(1.0 - target_sim**2) * ortho_v).astype(np.float32)
    close_v = close_v / np.linalg.norm(close_v)

    # Store reference images and embeddings
    img1 = ProductImage(
        sku_id=sku1.id, role="reference", split="reference",
        filename=f"d1_{time.time()}.jpg", file_path=f"d1_{time.time()}.jpg", file_hash=f"h1_{time.time()}",
        file_size_bytes=100, width=224, height=224, mime_type="image/jpeg", is_valid=True,
    )
    img2 = ProductImage(
        sku_id=sku2.id, role="reference", split="reference",
        filename=f"d2_{time.time()}.jpg", file_path=f"d2_{time.time()}.jpg", file_hash=f"h2_{time.time()}",
        file_size_bytes=100, width=224, height=224, mime_type="image/jpeg", is_valid=True,
    )
    db.add_all([img1, img2])
    db.commit()

    emb1 = Embedding(image_id=img1.id, sku_id=sku1.id, model_name=active_model.model_id, embedding_dim=dim, vector_bytes=base_v.tobytes(), is_active=True)
    emb2 = Embedding(image_id=img2.id, sku_id=sku2.id, model_name=active_model.model_id, embedding_dim=dim, vector_bytes=close_v.tobytes(), is_active=True)
    db.add_all([emb1, emb2])
    db.commit()
    invalidate_gallery_cache()

    # Query with base_v: Top 1 is sku1 (sim 1.0), Top 2 is sku2 (sim 0.995)
    # Margin delta is 0.005 < 0.08
    res = gallery_service.search_candidates(
        db,
        query_vector=base_v,
        similarity_threshold=0.60,
        margin_threshold=0.08,
    )

    # Margin gate MUST reject as ambiguous candidate
    assert res["is_unknown"] is True
    assert res["recognition_status"] in ("UNKNOWN", "AMBIGUOUS")
    assert "Ambiguous" in str(res["predicted_sku_name"]) or "margin" in str(res["rejection_reason"]).lower()


def test_model_activation_and_rollback_safety(db):
    """
    Rule 8: Verifies transactional activation to DINO, vector re-indexing,
    and instantaneous safe rollback to MobileNetV3 baseline.
    """
    initial_active = get_active_model(db, model_type="embedding")
    assert initial_active.model_id == "mobilenetv3-pretrained-baseline"

    # Step 1: Activate DINOv2 ViT-S
    act_res = activate_embedding_model(db, "dinov2-vits14-baseline")
    assert act_res["status"] in ["ACTIVATED", "ALREADY_ACTIVE"]

    current_active = get_active_model(db, model_type="embedding")
    assert current_active.model_id == "dinov2-vits14-baseline"
    assert current_active.embedding_dim == 384

    # Step 2: Rollback to MobileNetV3
    rb_res = activate_embedding_model(db, "mobilenetv3-pretrained-baseline")
    assert rb_res["status"] == "ACTIVATED"

    restored_active = get_active_model(db, model_type="embedding")
    assert restored_active.model_id == "mobilenetv3-pretrained-baseline"
    assert restored_active.embedding_dim == 960


def test_inference_latency_benchmark():
    """
    Measures CPU / GPU inference latency of Faster R-CNN detector + DINO embedder.
    """
    detector = BaselineDetector()
    embedder = BaselineEmbedder(model_id="dinov2-vits14-baseline")

    dummy_frame = Image.new("RGB", (640, 480), color=(100, 150, 200))

    # Warmup
    _ = detector.detect_and_crop(dummy_frame, allow_fallback=True, save_crops=False)
    _ = embedder.extract_embedding_from_crop(dummy_frame)

    # Measure Detector Latency
    t0 = time.time()
    detections, is_fallback, _ = detector.detect_and_crop(dummy_frame, allow_fallback=True, save_crops=False)
    det_ms = (time.time() - t0) * 1000

    # Measure DINO Embedder Latency
    t1 = time.time()
    _ = embedder.extract_embedding_from_crop(detections[0]["crop_img"])
    emb_ms = (time.time() - t1) * 1000

    print(f"\n[LATENCY REPORT] Detector: {det_ms:.1f}ms | DINO ViT-S Embedder: {emb_ms:.1f}ms | Total Pipeline: {det_ms + emb_ms:.1f}ms")
    assert det_ms < 1500  # CPU bounding limit
    assert emb_ms < 1000  # CPU bounding limit
