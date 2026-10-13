from datetime import datetime
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, Float
from app.core.database import Base


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String(100), unique=True, index=True, nullable=False)
    model_type = Column(String(50), default="embedding", nullable=False)  # 'embedding' or 'detector'
    architecture = Column(String(100), nullable=False)
    pretrained_source = Column(String(255), nullable=False)
    weight_version = Column(String(100), nullable=False)
    embedding_dim = Column(Integer, nullable=False)
    checkpoint_path = Column(String(512), nullable=True)
    hyperparameters_json = Column(Text, nullable=True)
    metrics_json = Column(Text, nullable=True)
    status = Column(String(50), default="ready", nullable=False)  # 'ready', 'training', 'failed', 'evaluating'
    is_active = Column(Boolean, default=False, nullable=False)
    license = Column(String(100), default="BSD-3-Clause", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
