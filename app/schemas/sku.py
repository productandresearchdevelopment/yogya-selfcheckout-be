from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict


class SKUBase(BaseModel):
    sku_code: str = Field(..., min_length=2, max_length=100, description="Unique SKU code")
    name: str = Field(..., min_length=2, max_length=255, description="Product Name")
    category: Optional[str] = Field(None, max_length=100, description="Product Category")
    description: Optional[str] = Field(None, description="Detailed product description")


class SKUCreate(SKUBase):
    pass


class SKUUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=2, max_length=255)
    category: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None


class ProductImageSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: str
    split: str
    filename: str
    file_path: str
    file_hash: str
    file_size_bytes: int
    width: int
    height: int
    blur_score: Optional[float] = None
    brightness_score: Optional[float] = None
    contrast_score: Optional[float] = None
    is_valid: bool
    validation_notes: Optional[str] = None
    created_at: datetime
    has_embedding: bool = False


class SKUResponse(SKUBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    is_active: bool
    created_at: datetime
    updated_at: datetime
    reference_image_count: int = 0
    training_image_count: int = 0
    total_image_count: int = 0
    reference_images: List[ProductImageSummary] = []
    training_images: List[ProductImageSummary] = []


class FewShotEnrollmentResponse(BaseModel):
    status: str
    sku_id: int
    sku_code: str
    name: str
    category: Optional[str] = None
    enrolled_images_count: int
    model_id: str
    embedding_dim: int
    message: str
    retrained: bool = False
    sku: SKUResponse

