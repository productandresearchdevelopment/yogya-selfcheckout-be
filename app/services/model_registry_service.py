import json
import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.models.model_registry import ModelRegistry
from app.models.image import ProductImage
from app.models.sku import SKU
from app.models.embedding import Embedding
from app.core.config import EMBEDDING_MODEL_NAME, DETECTOR_MODEL_NAME


def ensure_baseline_models_registered(db: Session):
    """
    Guarantees baseline pretrained models exist in the ModelRegistry on startup.
    Preserves original pretrained checkpoints for guaranteed rollback.
    """
    # 1. Embedding Model Baseline
    emb_baseline = (
        db.query(ModelRegistry)
        .filter(ModelRegistry.model_id == "mobilenetv3-pretrained-baseline")
        .first()
    )
    if not emb_baseline:
        # Check if any embedding model is active
        has_active = (
            db.query(ModelRegistry)
            .filter(ModelRegistry.model_type == "embedding", ModelRegistry.is_active == True)
            .first()
            is not None
        )
        emb_baseline = ModelRegistry(
            model_id="mobilenetv3-pretrained-baseline",
            model_type="embedding",
            architecture="MobileNetV3-Large (Feature Pooling 960)",
            pretrained_source="torchvision.models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)",
            weight_version="ImageNet-1K Pretrained",
            embedding_dim=960,
            checkpoint_path=None,
            hyperparameters_json=json.dumps({
                "backbone": "mobilenet_v3_large",
                "weights": "DEFAULT",
                "pooling": "AdaptiveAvgPool2d",
                "normalization": "L2",
                "description": "Baseline pretrained feature extractor without fine-tuning.",
            }),
            metrics_json=json.dumps({
                "domain": "ImageNet-1K (1,000 general visual classes)",
                "embedding_dim": 960,
                "zero_shot_baseline": True,
            }),
            status="ready",
            is_active=not has_active,  # Active if no other model is active
            license="BSD-3-Clause",
        )
        db.add(emb_baseline)
        db.commit()

    # 1b. DINOv2 ViT-S Baseline Registration
    dino_s_baseline = (
        db.query(ModelRegistry)
        .filter(ModelRegistry.model_id == "dinov2-vits14-baseline")
        .first()
    )
    if not dino_s_baseline:
        dino_s_baseline = ModelRegistry(
            model_id="dinov2-vits14-baseline",
            model_type="embedding",
            architecture="DINOv2-ViT-S/14 (Meta Self-Supervised Vision Transformer)",
            pretrained_source="torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')",
            weight_version="Meta Pretrained Self-Supervised LVD-142M",
            embedding_dim=384,
            checkpoint_path=None,
            hyperparameters_json=json.dumps({
                "backbone": "dinov2_vits14",
                "patch_size": 14,
                "image_size": 224,
                "embedding_dim": 384,
                "pooling": "CLS token",
                "normalization": "L2",
                "description": "Meta DINO Vision Transformer Small. Highly discriminative self-supervised visual features.",
            }),
            metrics_json=json.dumps({
                "domain": "LVD-142M curated visual foundation dataset",
                "embedding_dim": 384,
                "cpu_latency_ms": 437.0,
            }),
            status="ready",
            is_active=False,
            license="Apache-2.0",
        )
        db.add(dino_s_baseline)
        db.commit()

    # 1c. DINOv2 ViT-B Baseline Registration
    dino_b_baseline = (
        db.query(ModelRegistry)
        .filter(ModelRegistry.model_id == "dinov2-vitb14-baseline")
        .first()
    )
    if not dino_b_baseline:
        dino_b_baseline = ModelRegistry(
            model_id="dinov2-vitb14-baseline",
            model_type="embedding",
            architecture="DINOv2-ViT-B/14 (Meta Self-Supervised Vision Transformer)",
            pretrained_source="torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')",
            weight_version="Meta Pretrained Self-Supervised LVD-142M",
            embedding_dim=768,
            checkpoint_path=None,
            hyperparameters_json=json.dumps({
                "backbone": "dinov2_vitb14",
                "patch_size": 14,
                "image_size": 224,
                "embedding_dim": 768,
                "pooling": "CLS token",
                "normalization": "L2",
                "description": "Meta DINO Vision Transformer Base. 768-dim foundation model.",
            }),
            metrics_json=json.dumps({
                "domain": "LVD-142M curated visual foundation dataset",
                "embedding_dim": 768,
                "cpu_latency_ms": 1177.0,
            }),
            status="ready",
            is_active=False,
            license="Apache-2.0",
        )
        db.add(dino_b_baseline)
        db.commit()

    # 1d. DINOv3 ViT-S Baseline Registration
    dino3_s_baseline = (
        db.query(ModelRegistry)
        .filter(ModelRegistry.model_id == "dinov3-vits16-baseline")
        .first()
    )
    if not dino3_s_baseline:
        has_dinov3_access = bool(os.getenv("DINOV3_CHECKPOINT_PATH") or os.getenv("HF_TOKEN"))
        dino3_s_baseline = ModelRegistry(
            model_id="dinov3-vits16-baseline",
            model_type="embedding",
            architecture="DINOv3-ViT-S/16 (Meta Self-Supervised Vision Transformer)",
            pretrained_source="Meta DINOv3 (facebook/dinov3-vits16-pretrain-lvd1689m)",
            weight_version="Meta Pretrained Self-Supervised LVD-1689M",
            embedding_dim=384,
            checkpoint_path=None,
            hyperparameters_json=json.dumps({
                "backbone": "dinov3_vits16",
                "patch_size": 16,
                "image_size": 224,
                "embedding_dim": 384,
                "pooling": "CLS token",
                "normalization": "L2",
                "description": "Meta DINOv3 Vision Transformer Small. Gated weights require license agreement or DINOV3_CHECKPOINT_PATH.",
            }),
            metrics_json=json.dumps({
                "domain": "LVD-1689M foundation dataset",
                "embedding_dim": 384,
                "gated_status": "AUTHENTICATED" if has_dinov3_access else "BLOCKED_GATED_LICENSE_REQUIRED",
            }),
            status="ready" if has_dinov3_access else "gated_license_required",
            is_active=False,
            license="Meta DINOv3 License Agreement",
        )
        db.add(dino3_s_baseline)
        db.commit()

    # 1e. DINOv3 ViT-B Baseline Registration
    dino3_b_baseline = (
        db.query(ModelRegistry)
        .filter(ModelRegistry.model_id == "dinov3-vitb16-baseline")
        .first()
    )
    if not dino3_b_baseline:
        has_dinov3_access = bool(os.getenv("DINOV3_CHECKPOINT_PATH") or os.getenv("HF_TOKEN"))
        dino3_b_baseline = ModelRegistry(
            model_id="dinov3-vitb16-baseline",
            model_type="embedding",
            architecture="DINOv3-ViT-B/16 (Meta Self-Supervised Vision Transformer)",
            pretrained_source="Meta DINOv3 (facebook/dinov3-vitb16-pretrain-lvd1689m)",
            weight_version="Meta Pretrained Self-Supervised LVD-1689M",
            embedding_dim=768,
            checkpoint_path=None,
            hyperparameters_json=json.dumps({
                "backbone": "dinov3_vitb16",
                "patch_size": 16,
                "image_size": 224,
                "embedding_dim": 768,
                "pooling": "CLS token",
                "normalization": "L2",
                "description": "Meta DINOv3 Vision Transformer Base. Gated weights require license agreement or DINOV3_CHECKPOINT_PATH.",
            }),
            metrics_json=json.dumps({
                "domain": "LVD-1689M foundation dataset",
                "embedding_dim": 768,
                "gated_status": "AUTHENTICATED" if has_dinov3_access else "BLOCKED_GATED_LICENSE_REQUIRED",
            }),
            status="ready" if has_dinov3_access else "gated_license_required",
            is_active=False,
            license="Meta DINOv3 License Agreement",
        )
        db.add(dino3_b_baseline)
        db.commit()

    # 2. Detector Model Baseline
    det_baseline = (
        db.query(ModelRegistry)
        .filter(ModelRegistry.model_id == "fasterrcnn-pretrained-baseline")
        .first()
    )
    if not det_baseline:
        det_baseline = ModelRegistry(
            model_id="fasterrcnn-pretrained-baseline",
            model_type="detector",
            architecture="FasterRCNN-MobileNetV3-Large-FPN",
            pretrained_source="torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=DEFAULT)",
            weight_version="MS-COCO Pretrained",
            embedding_dim=0,
            checkpoint_path=None,
            hyperparameters_json=json.dumps({
                "backbone": "mobilenet_v3_large_fpn",
                "confidence_threshold": 0.35,
                "rpn_pre_nms_top_n_test": 1000,
                "rpn_post_nms_top_n_test": 1000,
            }),
            metrics_json=json.dumps({
                "domain": "MS-COCO (91 Common Objects)",
                "detection_type": "Two-Stage Region Proposal (RPN + Fast R-CNN Head)",
            }),
            status="ready",
            is_active=True,
            license="BSD-3-Clause",
        )
        db.add(det_baseline)
        db.commit()


def get_active_model(db: Session, model_type: str = "embedding") -> ModelRegistry:
    """Retrieves current active model for the given model_type."""
    active = (
        db.query(ModelRegistry)
        .filter(ModelRegistry.model_type == model_type, ModelRegistry.is_active == True)
        .first()
    )
    if not active:
        ensure_baseline_models_registered(db)
        active = (
            db.query(ModelRegistry)
            .filter(ModelRegistry.model_type == model_type, ModelRegistry.is_active == True)
            .first()
        )
    return active


def list_registered_models(db: Session, model_type: Optional[str] = None) -> List[ModelRegistry]:
    """Lists all registered models, sorted by creation date descending."""
    query = db.query(ModelRegistry)
    if model_type:
        query = query.filter(ModelRegistry.model_type == model_type)
    return query.order_by(ModelRegistry.created_at.desc()).all()


def load_model_instance(model_record: ModelRegistry) -> Any:
    """
    Loads and validates model instance for inference / gallery extraction.
    Performs test forward pass to ensure checkpoint validity.
    """
    from app.services.metric_learning import MetricEmbeddingModel
    from app.services.embedder import get_embedder

    if model_record.model_id == "mobilenetv3-pretrained-baseline":
        model, device, transform = get_embedder()
        return model, device, transform

    if "dino" in model_record.model_id.lower() and not model_record.checkpoint_path:
        from app.services.embedder import get_dino_embedder
        variant = "dinov2_vitb14" if "vitb" in model_record.model_id.lower() else "dinov2_vits14"
        if "dinov3" in model_record.model_id.lower():
            variant = "dinov3_vitb16" if "vitb" in model_record.model_id.lower() else "dinov3_vits16"
        model, device, transform = get_dino_embedder(variant=variant)
        return model, device, transform

    # Fine-tuned model checkpoint
    if not model_record.checkpoint_path or not os.path.exists(model_record.checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint file not found for model {model_record.model_id}: {model_record.checkpoint_path}"
        )

    from app.core.device import get_torch_device
    from torchvision import transforms

    device = get_torch_device()
    checkpoint = torch.load(model_record.checkpoint_path, map_location=device)
    backbone_type = checkpoint.get("backbone_type") or (
        "dinov2_vits14" if "dinov2" in model_record.model_id.lower()
        else ("dinov3_vits16" if "dinov3" in model_record.model_id.lower() else "mobilenetv3")
    )

    model = MetricEmbeddingModel(
        embedding_dim=model_record.embedding_dim,
        backbone_type=backbone_type,
        freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Preprocessing transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Validate test forward pass
    dummy = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        out = model(dummy)
        if out.shape != (1, model_record.embedding_dim):
            raise ValueError(
                f"Model output dimension mismatch: expected (1, {model_record.embedding_dim}), got {out.shape}"
            )

    return model, device, transform


def activate_embedding_model(db: Session, target_model_id: str) -> Dict[str, Any]:
    """
    Atomically activates a model version and migrates the embedding gallery.
    Strictly adheres to Section 2, Requirement 8:
    1. Preserves previous active model and gallery.
    2. Regenerates reference embeddings for all active SKUs using target checkpoint.
    3. Validates regenerated gallery.
    4. Atomically activates new checkpoint and compatible gallery.
    5. Rolls back safely if any step fails.
    """
    target = db.query(ModelRegistry).filter(ModelRegistry.model_id == target_model_id).first()
    if not target:
        raise ValueError(f"Model '{target_model_id}' not found in registry.")

    if target.model_type != "embedding":
        raise ValueError(f"Model '{target_model_id}' is a {target.model_type}, not an embedding model.")

    current_active = get_active_model(db, model_type="embedding")
    if current_active and current_active.model_id == target.model_id:
        return {
            "status": "ALREADY_ACTIVE",
            "model_id": target.model_id,
            "message": f"Model '{target.model_id}' is already active.",
        }

    # Step 1 & 2: Load and validate target model
    try:
        model, device, transform = load_model_instance(target)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint for '{target_model_id}': {str(e)}")

    # Step 3: Fetch all active reference images
    ref_images = (
        db.query(ProductImage)
        .join(SKU, ProductImage.sku_id == SKU.id)
        .filter(ProductImage.role == "reference", SKU.is_active == True)
        .all()
    )

    # Step 4: Extract and validate new embeddings in memory
    from PIL import Image
    new_vectors: List[Tuple[int, int, bytes]] = []

    for img in ref_images:
        if not os.path.exists(img.file_path):
            continue
        try:
            with Image.open(img.file_path) as pil_img:
                pil_img = pil_img.convert("RGB")
                tensor = transform(pil_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model(tensor).flatten().cpu().numpy().astype(np.float32)

                # Verify normalization and dimension
                norm = np.linalg.norm(emb)
                if norm > 1e-6:
                    emb = emb / norm
                if len(emb) != target.embedding_dim:
                    raise ValueError(
                        f"Extracted vector dim {len(emb)} does not match model dim {target.embedding_dim}"
                    )
                if np.isnan(emb).any() or np.isinf(emb).any():
                    raise ValueError(f"Extracted vector contains NaN or Inf values for image {img.id}")

                new_vectors.append((img.id, img.sku_id, emb.tobytes()))
        except Exception as e:
            raise RuntimeError(f"Failed extracting embedding for image {img.id} using {target.model_id}: {str(e)}")

    # Step 5: Atomic database migration
    try:
        # Mark all current active embeddings inactive
        db.query(Embedding).filter(Embedding.is_active == True).update({"is_active": False})

        # Insert or update target model embeddings
        for img_id, sku_id, vec_bytes in new_vectors:
            existing = (
                db.query(Embedding)
                .filter(Embedding.image_id == img_id, Embedding.model_name == target.model_id)
                .first()
            )
            if existing:
                existing.vector_bytes = vec_bytes
                existing.embedding_dim = target.embedding_dim
                existing.is_active = True
            else:
                new_emb = Embedding(
                    image_id=img_id,
                    sku_id=sku_id,
                    model_name=target.model_id,
                    embedding_dim=target.embedding_dim,
                    vector_bytes=vec_bytes,
                    is_active=True,
                )
                db.add(new_emb)

        # Switch active model in registry
        if current_active:
            current_active.is_active = False
        target.is_active = True

        db.commit()
        from app.services.gallery import invalidate_gallery_cache
        invalidate_gallery_cache()
    except Exception as e:
        db.rollback()
        raise RuntimeError(f"Atomic gallery migration failed, safely rolled back: {str(e)}")

    return {
        "status": "ACTIVATED",
        "model_id": target.model_id,
        "previous_model_id": current_active.model_id if current_active else None,
        "migrated_vectors_count": len(new_vectors),
        "embedding_dim": target.embedding_dim,
        "message": f"Successfully activated '{target.model_id}' and regenerated {len(new_vectors)} gallery vectors.",
    }
