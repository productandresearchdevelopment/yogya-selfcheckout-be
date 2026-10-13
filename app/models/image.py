from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from app.core.database import Base


class ProductImage(Base):
    __tablename__ = "product_images"

    id = Column(Integer, primary_key=True, index=True)
    sku_id = Column(Integer, ForeignKey("skus.id", ondelete="CASCADE"), nullable=True, index=True)
    role = Column(String(50), nullable=False, default="training")  # 'reference', 'training', 'validation', 'test', 'query'
    split = Column(String(50), nullable=False, default="unassigned")  # 'train', 'val', 'test', 'reference', 'unassigned'
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_hash = Column(String(64), index=True, nullable=False)  # SHA-256 for duplicate detection
    file_size_bytes = Column(Integer, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    mime_type = Column(String(50), nullable=False)
    blur_score = Column(Float, nullable=True)  # Laplacian variance (higher = sharper, <100 = blurry)
    brightness_score = Column(Float, nullable=True)  # Mean pixel brightness (0..255)
    contrast_score = Column(Float, nullable=True)  # RMS contrast / std dev
    is_valid = Column(Boolean, default=True, nullable=False)
    validation_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    sku = relationship("SKU", back_populates="images")
    annotations = relationship("Annotation", back_populates="image", cascade="all, delete-orphan")
    embeddings = relationship("Embedding", back_populates="image", cascade="all, delete-orphan")

    @property
    def embedding(self):
        """Active embedding for this image."""
        for emb in self.embeddings:
            if emb.is_active:
                return emb
        return None
