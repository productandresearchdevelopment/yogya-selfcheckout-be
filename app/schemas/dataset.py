from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict


class ImageQualityMetrics(BaseModel):
    width: int
    height: int
    file_size_bytes: int
    blur_score: float
    brightness_score: float
    contrast_score: float
    is_sharp: bool
    is_adequate_resolution: bool
    is_adequate_lighting: bool
    issues: List[str] = []


class AnnotationCreate(BaseModel):
    x_min: float = Field(..., ge=0.0, le=1.0)
    y_min: float = Field(..., ge=0.0, le=1.0)
    x_max: float = Field(..., ge=0.0, le=1.0)
    y_max: float = Field(..., ge=0.0, le=1.0)
    label: str = Field(..., min_length=1)
    sku_id: Optional[int] = None
    confidence: float = 1.0
    source: str = "manual"


class AnnotationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    image_id: int
    sku_id: Optional[int] = None
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    px_x: Optional[int] = None
    px_y: Optional[int] = None
    px_width: Optional[int] = None
    px_height: Optional[int] = None
    label: str
    confidence: float
    source: str
    created_at: datetime


class DatasetSplitUpdate(BaseModel):
    image_id: int
    split: str  # 'train', 'val', 'test', 'reference', 'unassigned'


class DatasetExportResponse(BaseModel):
    format: str
    total_images: int
    total_annotations: int
    splits: dict
    archive_path: Optional[str] = None
    message: str
