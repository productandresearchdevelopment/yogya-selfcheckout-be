import os
import json
import zipfile
import pytest
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.main import app
from app.core.database import SessionLocal
from app.services.gallery import (
    GalleryService,
    invalidate_gallery_cache,
)
from app.services.model_registry_service import get_active_model
from app.services.backup_service import (
    create_backup,
    verify_backup_integrity,
    restore_backup,
    cleanup_temporary_artifacts,
    BACKUPS_DIR,
)

client = TestClient(app)


@pytest.fixture
def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_system_readiness_endpoint():
    """Verify system readiness endpoint reports model readiness, gallery sync, and stats."""
    response = client.get("/api/v1/system/readiness")
    assert response.status_code == 200
    data = response.json()

    assert data["overall_status"] in ["READY", "DEGRADED", "ERROR"]
    assert data["application_health"] == "ONLINE"
    assert "active_models" in data
    assert "gallery_stats" in data
    assert "registered_active_skus" in data["gallery_stats"]
    assert "active_reference_vectors" in data["gallery_stats"]


def test_system_resources_endpoint():
    """Verify host resource monitor reports CPU, RAM, disk, and process RSS telemetry."""
    response = client.get("/api/v1/system/resources")
    assert response.status_code == 200
    data = response.json()

    # CPU metrics
    assert "cpu" in data
    assert data["cpu"]["cpu_cores"] > 0
    assert "cpu_percent" in data["cpu"]

    # RAM metrics
    assert "ram" in data
    assert data["ram"]["total_gb"] > 0
    assert data["ram"]["available_gb"] >= 0
    assert data["ram"]["process_rss_mb"] > 0

    # Disk metrics
    assert "disk" in data
    assert data["disk"]["total_gb"] > 0
    assert data["disk"]["free_gb"] > 0

    # Runtime
    assert "runtime" in data
    assert data["runtime"]["torch_threads"] >= 1


def test_in_memory_gallery_cache_and_blas_search(db_session: Session):
    """Verify that gallery cache builds a valid L2-normalized matrix and invalidates properly."""
    invalidate_gallery_cache()
    gallery_service = GalleryService()
    active_model = get_active_model(db_session, model_type="embedding")

    matrix, metadata = gallery_service._get_or_load_cache(
        db_session, active_model.model_id, active_model.embedding_dim
    )
    if matrix is not None:
        assert isinstance(matrix, np.ndarray)
        assert len(matrix.shape) == 2
        assert len(metadata) == matrix.shape[0]

        # Check L2 normalization (norms should be ~1.0)
        norms = np.linalg.norm(matrix, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-3)

        # Test vector search against top vector
        query_vec = matrix[0]
        match = gallery_service.search_candidates(db_session, query_vec, similarity_threshold=0.50)
        assert match["predicted_sku"] == metadata[0]["sku_code"]
        assert match["top_similarity_score"] >= 0.99  # Self-match cosine similarity ~ 1.0

    # Invalidate and ensure fresh reload works
    invalidate_gallery_cache()
    matrix_reloaded, _ = gallery_service._get_or_load_cache(
        db_session, active_model.model_id, active_model.embedding_dim
    )
    if matrix is not None:
        assert matrix_reloaded is not None
        assert matrix_reloaded.shape == matrix.shape


def test_backup_creation_and_sha256_manifest(db_session: Session):
    """Verify atomic backup creation and SHA-256 integrity manifest generation."""
    res = create_backup(db_session, label="unit_test_backup")
    assert res["status"] == "SUCCESS"
    assert "backup_path" in res
    assert os.path.exists(res["backup_path"])

    archive_path = Path(res["backup_path"])
    try:
        # Verify integrity of freshly created backup
        integrity = verify_backup_integrity(archive_path)
        assert integrity["valid"] is True
        assert integrity["verified_files"] == integrity["total_files"]
        assert len(integrity["errors"]) == 0
        assert "manifest" in integrity
        assert integrity["manifest"]["counts"]["skus"] >= 0
    finally:
        # Clean up test backup file
        if archive_path.exists():
            archive_path.unlink()


def test_backup_tamper_detection(db_session: Session):
    """Verify that modifying a file or manifest inside the archive fails integrity verification."""
    res = create_backup(db_session, label="tamper_test_backup")
    archive_path = Path(res["backup_path"])

    try:
        # Create a tampered copy
        tampered_path = archive_path.parent / "tampered_test.zip"
        with zipfile.ZipFile(archive_path, "r") as zf_in:
            with zipfile.ZipFile(tampered_path, "w") as zf_out:
                for item in zf_in.infolist():
                    data = zf_in.read(item.filename)
                    if item.filename == "studio.db":
                        # Tamper with byte content
                        data = data + b"\x00CORRUPT\x00"
                    zf_out.writestr(item, data)

        tamper_check = verify_backup_integrity(tampered_path)
        assert tamper_check["valid"] is False
        assert len(tamper_check["errors"]) > 0
        assert any("SHA-256 mismatch for studio.db" in err for err in tamper_check["errors"])
    finally:
        if archive_path.exists():
            archive_path.unlink()
        if tampered_path.exists():
            tampered_path.unlink()


def test_backup_dry_run_sandbox_restore(db_session: Session):
    """Verify isolated dry-run restoration extracts files into sandbox without clobbering live files."""
    res = create_backup(db_session, label="restore_test_backup")
    archive_path = Path(res["backup_path"])

    try:
        restore_res = restore_backup(archive_path, dry_run=True)
        assert restore_res["status"] == "RESTORED"
        assert restore_res["is_isolated_dry_run"] is True
        assert os.path.exists(restore_res["destination"])
        # Sandbox must contain restored studio.db
        assert os.path.exists(os.path.join(restore_res["destination"], "studio.db"))
    finally:
        if archive_path.exists():
            archive_path.unlink()


def test_artifact_cleanup_dry_run(db_session: Session):
    """Verify artifact cleanup dry-run identifies safe items without deleting protected active assets."""
    cleanup_res = cleanup_temporary_artifacts(db_session, retain_last_checkpoints=2, dry_run=True)
    assert cleanup_res["dry_run"] is True
    assert "protected_models" in cleanup_res
    assert len(cleanup_res["protected_models"]) > 0
    assert cleanup_res["purged_checkpoints_count"] == 0
    assert cleanup_res["purged_crops_count"] == 0


def test_backup_restore_api_validation():
    """Verify API error handling for non-existent or invalid backup filenames."""
    # 404 for non-existent archive
    res_404 = client.post("/api/v1/system/backup/restore?filename=non_existent_archive_123.zip&dry_run=true")
    assert res_404.status_code == 404

    # List backups returns list
    list_res = client.get("/api/v1/system/backup/list")
    assert list_res.status_code == 200
    assert isinstance(list_res.json(), list)
