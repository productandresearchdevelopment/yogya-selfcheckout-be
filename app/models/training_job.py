from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean
from app.core.database import Base


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(100), unique=True, index=True, nullable=False)
    model_id = Column(String(100), nullable=False)
    status = Column(String(50), default="QUEUED", nullable=False)  # 'QUEUED', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED', 'INSUFFICIENT_DATA'
    total_epochs = Column(Integer, default=5, nullable=False)
    current_epoch = Column(Integer, default=0, nullable=False)
    train_loss = Column(Float, nullable=True)
    val_loss = Column(Float, nullable=True)
    val_accuracy = Column(Float, nullable=True)
    history_json = Column(Text, nullable=True)  # List of epoch metrics dicts
    config_json = Column(Text, nullable=True)
    logs = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    checkpoint_path = Column(String(512), nullable=True)
    is_cancelled = Column(Boolean, default=False, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
