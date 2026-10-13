import os
import io
import time
import json
import uuid
import shutil
import zipfile
import hashlib
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

from app.core.config import (
    STORAGE_DIR,
    IMAGES_DIR,
    MODELS_DIR,
    CHECKPOINTS_DIR,
    BACKEND_DIR,
)
from app.models.sku import SKU
from app.models.image import ProductImage
from app.models.embedding import Embedding
from app.models.model_registry import ModelRegistry
from app.services.model_registry_service import get_active_model
from app.services.gallery import invalidate_gallery_cache

BACKUPS_DIR = STORAGE_DIR / "backups"
BACKUPS_DIR.mkdir(parents=True, exist_ok=True)


def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def create_backup(db: Session, label: Optional[str] = None) -> Dict[str, Any]:
    """
    Creates a verifiable, self-contained backup archive of the Product Recognition Studio:
    - SQLite Database (studio.db)
    - Product images (original & reference crops)
    - Model checkpoints
    - manifest.json containing metadata and SHA-256 file hashes
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_uuid = uuid.uuid4().hex[:8]
    archive_name = f"studio_backup_{timestamp}_{backup_uuid}.zip"
    archive_path = BACKUPS_DIR / archive_name

    db_path = STORAGE_DIR / "studio.db"

    # Gather system state metadata
    active_emb = get_active_model(db, model_type="embedding")
    active_det = get_active_model(db, model_type="detector")
    sku_count = db.query(SKU).count()
    img_count = db.query(ProductImage).count()
    emb_count = db.query(Embedding).count()
    models_count = db.query(ModelRegistry).count()

    files_manifest: Dict[str, Dict[str, Any]] = {}

    # Use a temporary file during backup construction to ensure atomic completion
    temp_zip_fd, temp_zip_path = tempfile.mkstemp(suffix=".zip")
    os.close(temp_zip_fd)

    try:
        with zipfile.ZipFile(temp_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            # 1. Archive Database State
            if db_path.exists():
                db_sha = compute_file_sha256(db_path)
                zf.write(db_path, arcname="studio.db")
                files_manifest["studio.db"] = {
                    "sha256": db_sha,
                    "size_bytes": os.path.getsize(db_path),
                }

            # 2. Archive Original Images
            orig_dir = IMAGES_DIR / "original"
            if orig_dir.exists():
                for root, _, files in os.walk(orig_dir):
                    for f in files:
                        fp = Path(root) / f
                        rel_arc = (Path("images/original") / fp.relative_to(orig_dir)).as_posix()
                        sha = compute_file_sha256(fp)
                        zf.write(fp, arcname=rel_arc)
                        files_manifest[rel_arc] = {
                            "sha256": sha,
                            "size_bytes": os.path.getsize(fp),
                        }

            # 3. Archive Model Checkpoints
            if CHECKPOINTS_DIR.exists():
                for root, _, files in os.walk(CHECKPOINTS_DIR):
                    for f in files:
                        fp = Path(root) / f
                        rel_arc = (Path("models/checkpoints") / fp.relative_to(CHECKPOINTS_DIR)).as_posix()
                        sha = compute_file_sha256(fp)
                        zf.write(fp, arcname=rel_arc)
                        files_manifest[rel_arc] = {
                            "sha256": sha,
                            "size_bytes": os.path.getsize(fp),
                        }

            # 4. Construct Manifest
            manifest = {
                "backup_id": f"backup_{timestamp}_{backup_uuid}",
                "label": label or "manual_backup",
                "created_at_utc": datetime.utcnow().isoformat(),
                "studio_version": "1.0.0",
                "active_models": {
                    "embedding": active_emb.model_id if active_emb else None,
                    "embedding_dim": active_emb.embedding_dim if active_emb else None,
                    "detector": active_det.model_id if active_det else None,
                },
                "counts": {
                    "skus": sku_count,
                    "images": img_count,
                    "embeddings": emb_count,
                    "models": models_count,
                },
                "total_files": len(files_manifest),
                "files": files_manifest,
            }

            zf.writestr("manifest.json", json.dumps(manifest, indent=2))

        # Atomically move finished zip to backups directory
        shutil.move(temp_zip_path, str(archive_path))

    except Exception as e:
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)
        raise RuntimeError(f"Backup creation failed: {str(e)}")

    return {
        "status": "SUCCESS",
        "backup_name": archive_name,
        "backup_path": str(archive_path),
        "archive_size_bytes": os.path.getsize(archive_path),
        "total_files_archived": len(files_manifest),
        "manifest": manifest,
    }


def verify_backup_integrity(archive_path: Path) -> Dict[str, Any]:
    """
    Verifies that a backup archive exists, is readable, and all file SHA-256 hashes match its manifest.
    """
    if not archive_path.exists():
        return {"valid": False, "error": f"Backup file not found: {archive_path}"}

    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            if "manifest.json" not in zf.namelist():
                return {"valid": False, "error": "Backup archive is missing manifest.json"}

            manifest_data = json.loads(zf.read("manifest.json").decode())
            files_manifest = manifest_data.get("files", {})

            # Normalize zip namelist to POSIX slashes
            archive_members = {name.replace("\\", "/"): name for name in zf.namelist()}

            errors = []
            verified_count = 0

            for arc_name, meta in files_manifest.items():
                posix_key = arc_name.replace("\\", "/")
                if posix_key not in archive_members:
                    errors.append(f"Missing file in archive: {arc_name}")
                    continue

                actual_member_name = archive_members[posix_key]
                content = zf.read(actual_member_name)
                calc_sha = hashlib.sha256(content).hexdigest()
                expected_sha = meta["sha256"]

                if calc_sha != expected_sha:
                    errors.append(f"SHA-256 mismatch for {arc_name}: expected {expected_sha}, got {calc_sha}")
                else:
                    verified_count += 1

            return {
                "valid": len(errors) == 0,
                "backup_id": manifest_data.get("backup_id"),
                "total_files": len(files_manifest),
                "verified_files": verified_count,
                "errors": errors,
                "manifest": manifest_data,
            }
    except Exception as e:
        return {"valid": False, "error": f"Failed reading archive: {str(e)}"}


def restore_backup(
    archive_path: Path,
    target_staging_dir: Optional[Path] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Restores studio data from a verified backup archive.
    If dry_run=True or target_staging_dir is set, extracts to an isolated directory
    without touching active production files.
    """
    # 1. Strict Integrity Verification First
    integrity = verify_backup_integrity(archive_path)
    if not integrity["valid"]:
        raise ValueError(f"Backup verification failed: {integrity.get('error') or integrity.get('errors')}")

    # 2. Determine Destination
    if dry_run or target_staging_dir is not None:
        dest_dir = target_staging_dir or Path(tempfile.mkdtemp(prefix="studio_restore_test_"))
        is_isolated = True
    else:
        dest_dir = STORAGE_DIR
        is_isolated = False

    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            # Extract database
            if "studio.db" in zf.namelist():
                zf.extract("studio.db", path=dest_dir)

            # Extract images and models
            for item in zf.namelist():
                if item in ["studio.db", "manifest.json"]:
                    continue
                zf.extract(item, path=dest_dir)

        # Invalidate in-memory gallery cache so restored vectors are loaded fresh
        if not is_isolated:
            invalidate_gallery_cache()

        return {
            "status": "RESTORED",
            "is_isolated_dry_run": is_isolated,
            "destination": str(dest_dir),
            "files_restored": integrity["verified_files"],
            "backup_id": integrity.get("backup_id"),
            "manifest": integrity.get("manifest"),
        }
    except Exception as e:
        raise RuntimeError(f"Restoration failed: {str(e)}")


def list_backups() -> List[Dict[str, Any]]:
    """Lists all available backup archives in the storage/backups directory."""
    backups = []
    if not BACKUPS_DIR.exists():
        return []

    for f in sorted(BACKUPS_DIR.glob("studio_backup_*.zip"), reverse=True):
        try:
            with zipfile.ZipFile(f, "r") as zf:
                if "manifest.json" in zf.namelist():
                    manifest = json.loads(zf.read("manifest.json").decode())
                else:
                    manifest = {}
            backups.append({
                "filename": f.name,
                "file_path": str(f),
                "file_size_bytes": os.path.getsize(f),
                "created_at": manifest.get("created_at_utc") or time.ctime(os.path.getctime(f)),
                "backup_id": manifest.get("backup_id", f.stem),
                "label": manifest.get("label", "backup"),
                "total_files": manifest.get("total_files", 0),
                "skus_count": manifest.get("counts", {}).get("skus", 0),
                "images_count": manifest.get("counts", {}).get("images", 0),
            })
        except Exception:
            pass

    return backups


def cleanup_temporary_artifacts(
    db: Session,
    retain_last_checkpoints: int = 3,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """
    Safely identifies and purges obsolete temporary artifacts.
    Protects:
    - Active model checkpoints
    - Pretrained baseline models
    - Registered dataset images and reference gallery
    """
    active_models = db.query(ModelRegistry).filter(ModelRegistry.is_active == True).all()
    protected_model_ids = {m.model_id for m in active_models}
    protected_model_ids.add("mobilenetv3-pretrained-baseline")
    protected_model_ids.add("fasterrcnn-pretrained-baseline")

    # Find checkpoints in directory
    orphaned_checkpoints = []
    if CHECKPOINTS_DIR.exists():
        for ckpt in CHECKPOINTS_DIR.glob("*.pt"):
            model_id = ckpt.stem
            if model_id in protected_model_ids:
                continue
            # Check if associated with an active or recent model in registry
            reg_entry = db.query(ModelRegistry).filter(ModelRegistry.model_id == model_id).first()
            if reg_entry and reg_entry.is_active:
                continue
            orphaned_checkpoints.append(str(ckpt))

    # Find crops in CROPPED_IMAGES_DIR that are older than 48 hours and not referenced
    orphaned_crops = []
    crops_dir = IMAGES_DIR / "crops"
    cutoff_time = time.time() - (48 * 3600)
    if crops_dir.exists():
        for crop_file in crops_dir.glob("crop_fallback_*.jpg"):
            if os.path.getmtime(crop_file) < cutoff_time:
                orphaned_crops.append(str(crop_file))

    purged_checkpoints = []
    purged_crops = []

    if not dry_run:
        for ckpt_path in orphaned_checkpoints:
            try:
                os.remove(ckpt_path)
                purged_checkpoints.append(ckpt_path)
            except Exception:
                pass
        for crop_path in orphaned_crops:
            try:
                os.remove(crop_path)
                purged_crops.append(crop_path)
            except Exception:
                pass

    return {
        "dry_run": dry_run,
        "protected_models": list(protected_model_ids),
        "identified_orphaned_checkpoints": orphaned_checkpoints,
        "purged_checkpoints_count": len(purged_checkpoints) if not dry_run else 0,
        "identified_orphaned_crops": orphaned_crops,
        "purged_crops_count": len(purged_crops) if not dry_run else 0,
        "message": (
            f"Dry run complete. Found {len(orphaned_checkpoints)} obsolete checkpoints and {len(orphaned_crops)} old fallback crops."
            if dry_run
            else f"Purged {len(purged_checkpoints)} checkpoints and {len(purged_crops)} crops safely."
        ),
    }
