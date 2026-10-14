"""
Database & Storage Reset Script for YOGYA Self-Checkout SKU Recognition.
Completely clears all sample/seeded SKUs, product images, embeddings, annotations,
hard negatives, logs, and temporary checkpoint files while preserving baseline model definitions.
"""

import os
import shutil
from pathlib import Path
from sqlalchemy import text

# Setup paths
BACKEND_DIR = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(BACKEND_DIR))

from app.core.database import SessionLocal, engine
from app.core.config import ORIGINAL_IMAGES_DIR, CROPPED_IMAGES_DIR, CHECKPOINTS_DIR, EXPORTS_DIR, EMBEDDING_MODEL_NAME
from app.services.gallery import invalidate_gallery_cache
from app.services.model_registry_service import ensure_baseline_models_registered

def reset_database_and_storage():
    print("=" * 60)
    print("STARTING COMPLETE DATABASE & SAMPLE DATASET RESET")
    print("=" * 60)

    db = SessionLocal()
    try:
        # 1. Truncate all SKU, image, embedding, annotation, log, and job tables
        print("[1/5] Truncating PostgreSQL tables (skus, images, embeddings, logs, jobs)...")
        tables_to_truncate = [
            "recognition_logs",
            "annotations",
            "hard_negatives",
            "embeddings",
            "product_images",
            "skus",
            "training_jobs",
        ]
        truncate_sql = f"TRUNCATE TABLE {', '.join(tables_to_truncate)} CASCADE;"
        db.execute(text(truncate_sql))
        db.commit()
        print("      PostgreSQL data tables truncated successfully.")

        # 2. Clean up non-baseline models from model_registry
        print("[2/5] Cleaning up non-baseline model registry entries...")
        delete_models_sql = """
            DELETE FROM model_registry
            WHERE model_id NOT IN (
                'fasterrcnn-pretrained-baseline',
                'mobilenetv3-pretrained-baseline',
                'dinov2-vits14-baseline',
                'dinov2-vitb14-baseline',
                'dinov3-vits16-baseline',
                'dinov3-vitb16-baseline'
            );
        """
        db.execute(text(delete_models_sql))
        db.commit()

        # Re-verify baselines exist
        ensure_baseline_models_registered(db)

        # Ensure active baselines
        db.execute(text("UPDATE model_registry SET is_active = FALSE WHERE model_type = 'embedding';"))
        db.execute(text(f"UPDATE model_registry SET is_active = TRUE WHERE model_id = '{EMBEDDING_MODEL_NAME}';"))
        db.execute(text("UPDATE model_registry SET is_active = TRUE WHERE model_id = 'fasterrcnn-pretrained-baseline';"))
        db.commit()
        print(f"      Model registry reset to standard baseline models (active embedding: {EMBEDDING_MODEL_NAME}).")

        # 3. Clean up storage files (original images, crops, exports, checkpoints)
        print("[3/5] Cleaning storage directories...")
        
        # Clean original images
        deleted_orig = 0
        if ORIGINAL_IMAGES_DIR.exists():
            for item in ORIGINAL_IMAGES_DIR.iterdir():
                if item.is_file():
                    item.unlink()
                    deleted_orig += 1
                elif item.is_dir():
                    shutil.rmtree(item)
                    deleted_orig += 1
        print(f"      Removed {deleted_orig} files from storage/images/original/")

        # Clean cropped images
        deleted_crops = 0
        if CROPPED_IMAGES_DIR.exists():
            for item in CROPPED_IMAGES_DIR.iterdir():
                if item.is_file():
                    item.unlink()
                    deleted_crops += 1
                elif item.is_dir():
                    shutil.rmtree(item)
                    deleted_crops += 1
        print(f"      Removed {deleted_crops} files from storage/images/crops/")

        # Clean exports
        deleted_exports = 0
        if EXPORTS_DIR.exists():
            for item in EXPORTS_DIR.iterdir():
                if item.is_file():
                    item.unlink()
                    deleted_exports += 1
                elif item.is_dir():
                    shutil.rmtree(item)
                    deleted_exports += 1
        print(f"      Removed {deleted_exports} files/folders from storage/exports/")

        # Clean test/job checkpoints
        deleted_checkpoints = 0
        if CHECKPOINTS_DIR.exists():
            for item in CHECKPOINTS_DIR.iterdir():
                if item.is_dir() and (item.name.startswith("job_") or item.name.startswith("test_")):
                    shutil.rmtree(item)
                    deleted_checkpoints += 1
        print(f"      Removed {deleted_checkpoints} temporary checkpoint folders from storage/models/checkpoints/")

        # Reset legacy SQLite if exists
        sqlite_file = BACKEND_DIR / "storage" / "studio.db"
        if sqlite_file.exists():
            sqlite_file.unlink()
            print("      Removed legacy storage/studio.db file.")

        # 4. Invalidate Gallery in-memory cache
        print("[4/5] Invalidating in-memory recognition gallery cache...")
        invalidate_gallery_cache()
        print("      Gallery cache cleared.")

        # 5. Verify final counts
        print("[5/5] Verifying clean state...")
        counts = {}
        for t in tables_to_truncate + ["model_registry"]:
            cnt = db.execute(text(f"SELECT count(*) FROM {t}")).scalar()
            counts[t] = cnt
        
        print("\n" + "=" * 60)
        print("RESET SUMMARY - FINAL DATABASE ROW COUNTS:")
        print("=" * 60)
        for tbl, count in counts.items():
            print(f"  {tbl:<25}: {count}")
        print("=" * 60)
        print("DATABASE & STORAGE SUCCESSFULLY RESET AND READY FOR FRESH TESTING!")
        print("=" * 60)

    except Exception as e:
        db.rollback()
        print(f"\n[ERROR] Reset failed: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    reset_database_and_storage()
