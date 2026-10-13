from datetime import datetime
from sqlalchemy import Column, Integer, String, LargeBinary, Boolean, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from app.core.database import Base


class Embedding(Base):
    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("product_images.id", ondelete="CASCADE"), nullable=False, index=True)
    sku_id = Column(Integer, ForeignKey("skus.id", ondelete="CASCADE"), nullable=False, index=True)
    model_name = Column(String(100), nullable=False, index=True)
    embedding_dim = Column(Integer, nullable=False)
    vector_bytes = Column(LargeBinary, nullable=False)  # Serialized float32 numpy vector
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    image = relationship("ProductImage", back_populates="embeddings")
    sku = relationship("SKU", back_populates="embeddings")

    __table_args__ = (
        UniqueConstraint("image_id", "model_name", name="uq_image_model_embedding"),
    )
