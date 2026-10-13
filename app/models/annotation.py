from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.core.database import Base


class Annotation(Base):
    __tablename__ = "annotations"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("product_images.id", ondelete="CASCADE"), nullable=False, index=True)
    sku_id = Column(Integer, ForeignKey("skus.id", ondelete="SET NULL"), nullable=True, index=True)

    # Normalized coordinates [0.0, 1.0] relative to image dimensions
    x_min = Column(Float, nullable=False)
    y_min = Column(Float, nullable=False)
    x_max = Column(Float, nullable=False)
    y_max = Column(Float, nullable=False)

    # Pixel coordinates
    px_x = Column(Integer, nullable=True)
    px_y = Column(Integer, nullable=True)
    px_width = Column(Integer, nullable=True)
    px_height = Column(Integer, nullable=True)

    label = Column(String(100), nullable=False)
    confidence = Column(Float, default=1.0, nullable=False)
    source = Column(String(50), default="manual", nullable=False)  # 'manual', 'detector_baseline'
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    image = relationship("ProductImage", back_populates="annotations")
    sku = relationship("SKU", back_populates="annotations")
