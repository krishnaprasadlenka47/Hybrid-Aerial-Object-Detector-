from datetime import datetime
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionItem(BaseModel):
    category_id: int
    category_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBox


class DetectionResponse(BaseModel):
    filename: str
    image_width: int
    image_height: int
    total_detections: int
    detections: list[DetectionItem]
    inference_time_ms: float
    model_version: str

    model_config = {"from_attributes": True}


class DetectionLogOut(DetectionResponse):
    id: int
    created_at: datetime


class StatsResponse(BaseModel):
    total_images_processed: int
    total_objects_detected: int
    avg_detections_per_image: float
    avg_inference_time_ms: float
    category_counts: dict[str, int]
    recent_logs: list[DetectionLogOut]


class MetricsLogRequest(BaseModel):
    epoch: int
    map_50: float | None = None
    map_75: float | None = None
    map_overall: float | None = None
    loss_cls: float | None = None
    loss_bbox: float | None = None
    loss_total: float | None = None
    notes: str | None = None
