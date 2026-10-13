import os
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.core.database import get_db
from app.core.config import (
    STORAGE_DIR,
    ORIGINAL_IMAGES_DIR,
    CROPPED_IMAGES_DIR,
    DATASETS_DIR,
    MODELS_DIR,
    EXPORTS_DIR,
    DETECTOR_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
    DEFAULT_DETECTION_CONFIDENCE_THRESHOLD,
    DEFAULT_SIMILARITY_THRESHOLD,
)
from app.core.device import get_system_environment_info
from app.models.sku import SKU
from app.models.image import ProductImage
from app.models.annotation import Annotation
from app.models.embedding import Embedding
from app.models.recognition_log import RecognitionLog
from app.schemas.system import SystemHealthResponse, DashboardStatsResponse

router = APIRouter(prefix="/system", tags=["System"])


@router.get("/health", response_model=SystemHealthResponse)
def check_health(db: Session = Depends(get_db)):
    """Health check endpoint showing hardware acceleration, models, and DB status."""
    env_info = get_system_environment_info()

    db_ok = True
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        db_ok = False

    storage_paths = {
        "base_storage": str(STORAGE_DIR),
        "original_images": str(ORIGINAL_IMAGES_DIR),
        "crops": str(CROPPED_IMAGES_DIR),
        "datasets": str(DATASETS_DIR),
        "models": str(MODELS_DIR),
        "exports": str(EXPORTS_DIR),
    }

    model_versions = {
        "detector_backbone": DETECTOR_MODEL_NAME,
        "detector_weights": "FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT (COCO Pretrained)",
        "embedding_backbone": EMBEDDING_MODEL_NAME,
        "embedding_weights": "MobileNet_V3_Large_Weights.DEFAULT (ImageNet Pretrained)",
        "vector_search_engine": "NumPy Cosine Similarity (In-memory normalized dot-product)",
    }

    return SystemHealthResponse(
        status="HEALTHY" if db_ok else "DEGRADED",
        torch_available=env_info["torch_available"],
        torch_version=env_info["torch_version"],
        device=env_info["active_device"],
        cuda_available=env_info["cuda_available"],
        cuda_device_name=env_info["cuda_device_name"],
        mps_available=env_info["mps_available"],
        cpu_cores=env_info["cpu_count"],
        storage_paths=storage_paths,
        model_versions=model_versions,
        database_connected=db_ok,
    )


@router.get("/dashboard", response_model=DashboardStatsResponse)
def get_dashboard_stats(db: Session = Depends(get_db)):
    """Aggregated stats for the Dashboard page."""
    env_info = get_system_environment_info()

    sku_count = db.query(SKU).count()
    active_sku_count = db.query(SKU).filter(SKU.is_active == True).count()
    image_count = db.query(ProductImage).count()
    ref_count = db.query(ProductImage).filter(ProductImage.role == "reference").count()
    train_count = db.query(ProductImage).filter(ProductImage.role != "reference").count()
    annotation_count = db.query(Annotation).count()
    embedding_count = db.query(Embedding).count()
    recent_recognitions_count = db.query(RecognitionLog).count()

    recent_runs = (
        db.query(RecognitionLog)
        .order_by(RecognitionLog.created_at.desc())
        .limit(5)
        .all()
    )

    recent_runs_data = [
        {
            "id": r.id,
            "query_image_name": r.query_image_name,
            "num_detections": r.num_detections,
            "final_status": r.final_status,
            "top_predicted_sku": r.top_predicted_sku,
            "top_similarity": r.top_similarity,
            "duration_ms": r.duration_ms,
            "created_at": r.created_at.isoformat(),
        }
        for r in recent_runs
    ]

    return DashboardStatsResponse(
        sku_count=sku_count,
        active_sku_count=active_sku_count,
        image_count=image_count,
        reference_image_count=ref_count,
        training_image_count=train_count,
        annotation_count=annotation_count,
        embedding_count=embedding_count,
        recent_recognitions_count=recent_recognitions_count,
        device_info=env_info["active_device"],
        active_detector=DETECTOR_MODEL_NAME,
        active_embedder=EMBEDDING_MODEL_NAME,
        recent_runs=recent_runs_data,
    )


@router.get("/settings")
def get_settings():
    """Returns application configuration settings."""
    env_info = get_system_environment_info()
    return {
        "storage": {
            "storage_dir": str(STORAGE_DIR),
            "original_images": str(ORIGINAL_IMAGES_DIR),
            "crops": str(CROPPED_IMAGES_DIR),
            "datasets": str(DATASETS_DIR),
            "exports": str(EXPORTS_DIR),
        },
        "device": env_info,
        "models": {
            "detector_model": DETECTOR_MODEL_NAME,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "default_detection_threshold": DEFAULT_DETECTION_CONFIDENCE_THRESHOLD,
            "default_similarity_threshold": DEFAULT_SIMILARITY_THRESHOLD,
        },
        "mode": "local-first",
    }


# ===========================================================================
# SECTION 5: READINESS, RESOURCE MONITORING, BACKUP & RECOVERY
# ===========================================================================

@router.get("/readiness")
def get_system_readiness(db: Session = Depends(get_db)):
    """
    Distinguishes basic application health from model readiness and gallery compatibility
    per Section 5 Requirement 9.
    """
    from app.services.model_registry_service import get_active_model

    active_emb = get_active_model(db, model_type="embedding")
    active_det = get_active_model(db, model_type="detector")

    # Check gallery synchronization
    active_ref_embeddings = (
        db.query(Embedding)
        .filter(Embedding.model_name == active_emb.model_id, Embedding.is_active == True)
        .count()
    )
    registered_skus = db.query(SKU).filter(SKU.is_active == True).count()

    is_model_ready = (active_emb is not None and active_det is not None)
    is_gallery_synced = (active_ref_embeddings > 0 or registered_skus == 0)

    overall_status = "READY" if (is_model_ready and is_gallery_synced) else "DEGRADED"

    return {
        "overall_status": overall_status,
        "application_health": "ONLINE",
        "model_readiness": "READY" if is_model_ready else "UNAVAILABLE",
        "gallery_synchronization": "SYNCHRONIZED" if is_gallery_synced else "NEEDS_REINDEX",
        "active_models": {
            "embedding_model": active_emb.model_id if active_emb else None,
            "embedding_dim": active_emb.embedding_dim if active_emb else None,
            "detector_model": active_det.model_id if active_det else None,
        },
        "gallery_stats": {
            "registered_active_skus": registered_skus,
            "active_reference_vectors": active_ref_embeddings,
        },
    }


@router.get("/resources")
def get_resource_usage():
    """
    Measures CPU, RAM, PyTorch threads, and storage constraints
    per Section 5 Requirement 5.
    """
    import psutil
    import shutil

    process = psutil.Process()
    mem = psutil.virtual_memory()
    disk = shutil.disk_usage(STORAGE_DIR)

    return {
        "cpu": {
            "cpu_cores": psutil.cpu_count(logical=True),
            "physical_cores": psutil.cpu_count(logical=False),
            "cpu_percent": psutil.cpu_percent(interval=0.1),
        },
        "ram": {
            "total_gb": round(mem.total / (1024 ** 3), 2),
            "available_gb": round(mem.available / (1024 ** 3), 2),
            "used_gb": round(mem.used / (1024 ** 3), 2),
            "percent": mem.percent,
            "process_rss_mb": round(process.memory_info().rss / (1024 * 1024), 2),
        },
        "disk": {
            "storage_path": str(STORAGE_DIR),
            "total_gb": round(disk.total / (1024 ** 3), 2),
            "free_gb": round(disk.free / (1024 ** 3), 2),
            "percent_used": round((disk.used / disk.total) * 100, 1),
        },
        "runtime": {
            "torch_threads": import_torch_threads(),
            "device": get_system_environment_info()["active_device"],
        },
    }


def import_torch_threads():
    try:
        import torch
        return torch.get_num_threads()
    except Exception:
        return 1


@router.post("/backup/create")
def trigger_backup(
    label: str = "manual_backup",
    db: Session = Depends(get_db),
):
    """
    Creates an atomic, verifiable ZIP backup archive containing database, images,
    model checkpoints, and SHA-256 manifest.
    """
    from app.services.backup_service import create_backup
    try:
        res = create_backup(db, label=label)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backup creation failed: {str(e)}")


@router.get("/backup/list")
def list_backups_endpoint():
    """Lists all available backup archives with manifest metadata."""
    from app.services.backup_service import list_backups
    return list_backups()


@router.post("/backup/restore")
def restore_backup_endpoint(
    filename: str,
    dry_run: bool = True,
):
    """
    Restores studio state from a verified backup archive.
    Defaults to dry_run=True (staging extraction and verification without touching active files).
    """
    from app.services.backup_service import restore_backup, BACKUPS_DIR
    archive_path = BACKUPS_DIR / filename
    if not archive_path.exists():
        raise HTTPException(status_code=404, detail=f"Backup file '{filename}' not found.")

    try:
        res = restore_backup(archive_path, dry_run=dry_run)
        return res
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Restoration failed: {str(e)}")


@router.post("/cleanup")
def cleanup_artifacts_endpoint(
    retain_checkpoints: int = 3,
    dry_run: bool = True,
    db: Session = Depends(get_db),
):
    """
    Safely purges obsolete temporary artifacts and failed checkpoints while strictly
    protecting active models and dataset reference images.
    """
    from app.services.backup_service import cleanup_temporary_artifacts
    return cleanup_temporary_artifacts(db, retain_last_checkpoints=retain_checkpoints, dry_run=dry_run)
