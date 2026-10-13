from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Text, DateTime
from app.core.database import Base


class RecognitionLog(Base):
    __tablename__ = "recognition_logs"

    id = Column(Integer, primary_key=True, index=True)
    query_image_name = Column(String(255), nullable=False)
    query_image_path = Column(String(512), nullable=False)
    num_detections = Column(Integer, default=0, nullable=False)
    final_status = Column(String(50), default="UNKNOWN", nullable=False)  # 'MATCHED', 'UNKNOWN', 'NO_OBJECT_DETECTED'
    top_predicted_sku = Column(String(100), nullable=True)
    top_similarity = Column(Float, nullable=True)
    detector_model = Column(String(100), nullable=False)
    embedding_model = Column(String(100), nullable=False)
    results_json = Column(Text, nullable=False)  # Full JSON details
    duration_ms = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
