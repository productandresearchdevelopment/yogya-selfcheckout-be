import json
import uuid
import threading
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from sqlalchemy.orm import Session

from app.core.database import get_db, SessionLocal
from app.models.training_job import TrainingJob
from app.schemas.training import (
    DatasetReadinessResponse,
    StartTrainingRequest,
    TrainingJobResponse,
)
from app.services.readiness_service import (
    generate_dataset_readiness_report,
    seed_sample_training_dataset,
)
from app.services.metric_learning import (
    execute_metric_training_job,
    ACTIVE_CANCELLATIONS,
    TRAINING_MUTEX,
)

router = APIRouter(prefix="/training", tags=["Training & Metric Learning"])


@router.get("/readiness", response_model=DatasetReadinessResponse)
def get_readiness_report(db: Session = Depends(get_db)):
    """Audits dataset sufficiency for metric learning without fabricating readiness."""
    report = generate_dataset_readiness_report(db)
    return report


@router.post("/seed-samples")
def seed_dataset_samples(db: Session = Depends(get_db)):
    """
    Disabled in DINOv3 Few-Shot Architecture.
    Pretrained foundation model operates directly on enrolled reference photos without synthetic dataset generation.
    """
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Synthetic training dataset seeding is disabled. DINOv3 Few-Shot architecture uses genuine reference photos directly without retraining.",
    )


@router.post("/jobs", response_model=TrainingJobResponse, status_code=status.HTTP_201_CREATED)
def start_training_job(
    req: StartTrainingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Initiates asynchronous deep metric learning training job.
    Enforces resource concurrency lock and dataset readiness.
    """
    # 1. Check if another job is currently running
    if TRAINING_MUTEX.locked():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A model training job is currently running. Please wait for it to finish or cancel it.",
        )

    # 2. Check dataset readiness
    readiness = generate_dataset_readiness_report(db)
    if not readiness["is_ready"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Dataset is insufficient for deep metric learning.",
                "blockers": readiness["blockers"],
                "recommendations": readiness["recommendations"],
            },
        )

    # 3. Create TrainingJob DB record
    job_id = f"job_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    config_dict = req.model_dump()

    new_job = TrainingJob(
        job_id=job_id,
        model_id=f"{req.backbone_type}-metric-{job_id}",
        status="QUEUED",
        total_epochs=req.epochs,
        current_epoch=0,
        config_json=json.dumps(config_dict),
        logs=f"Job {job_id} queued for execution.\nConfig: {json.dumps(config_dict)}",
    )
    db.add(new_job)
    db.commit()
    db.refresh(new_job)

    # 4. Dispatch background thread
    thread = threading.Thread(
        target=execute_metric_training_job,
        args=(job_id, config_dict, SessionLocal),
        daemon=True,
    )
    thread.start()

    return _format_job_response(new_job)


@router.get("/jobs", response_model=List[TrainingJobResponse])
def list_training_jobs(db: Session = Depends(get_db)):
    """Lists all training jobs, ordered by creation date descending."""
    jobs = db.query(TrainingJob).order_by(TrainingJob.created_at.desc()).all()
    return [_format_job_response(j) for j in jobs]


@router.get("/jobs/{job_id}", response_model=TrainingJobResponse)
def get_training_job(job_id: str, db: Session = Depends(get_db)):
    """Fetches real-time status, epoch progress, and loss metrics for a specific training job."""
    job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Training job '{job_id}' not found.")
    return _format_job_response(job)


@router.post("/jobs/{job_id}/cancel")
def cancel_training_job(job_id: str, db: Session = Depends(get_db)):
    """Signals safe cancellation of an active training job."""
    job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Training job '{job_id}' not found.")

    if job.status not in ("QUEUED", "RUNNING"):
        return {"status": job.status, "message": f"Job is already {job.status}."}

    ACTIVE_CANCELLATIONS[job_id] = True
    job.is_cancelled = True
    db.commit()

    return {"status": "CANCELLING", "message": f"Cancellation requested for job '{job_id}'."}


def _format_job_response(job: TrainingJob) -> TrainingJobResponse:
    history = []
    if job.history_json:
        try:
            history = json.loads(job.history_json)
        except Exception:
            pass

    return TrainingJobResponse(
        job_id=job.job_id,
        model_id=job.model_id,
        status=job.status,
        total_epochs=job.total_epochs,
        current_epoch=job.current_epoch,
        train_loss=job.train_loss,
        val_loss=job.val_loss,
        val_accuracy=job.val_accuracy,
        history=history,
        logs=job.logs,
        error_message=job.error_message,
        checkpoint_path=job.checkpoint_path,
        started_at=str(job.started_at) if job.started_at else None,
        completed_at=str(job.completed_at) if job.completed_at else None,
        created_at=str(job.created_at),
    )
