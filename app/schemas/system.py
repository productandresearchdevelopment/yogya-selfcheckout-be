from typing import Dict, Any, List
from pydantic import BaseModel


class SystemHealthResponse(BaseModel):
    status: str
    torch_available: bool
    torch_version: str | None
    device: str
    cuda_available: bool
    cuda_device_name: str | None
    mps_available: bool
    cpu_cores: int
    storage_paths: Dict[str, str]
    model_versions: Dict[str, str]
    database_connected: bool


class DashboardStatsResponse(BaseModel):
    sku_count: int
    active_sku_count: int
    image_count: int
    reference_image_count: int
    training_image_count: int
    annotation_count: int
    embedding_count: int
    recent_recognitions_count: int
    device_info: str
    active_detector: str
    active_embedder: str
    recent_runs: List[Dict[str, Any]]
