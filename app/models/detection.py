from datetime import datetime
from sqlalchemy import String, Float, Integer, DateTime, JSON, Text, func
from sqlalchemy.orm import Mapped, mapped_column
from app.db.database import Base


class DetectionLog(Base):
    __tablename__ = "detection_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    image_width: Mapped[int] = mapped_column(Integer, nullable=False)
    image_height: Mapped[int] = mapped_column(Integer, nullable=False)
    total_detections: Mapped[int] = mapped_column(Integer, nullable=False)
    detections: Mapped[dict] = mapped_column(JSON, nullable=False)
    inference_time_ms: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False, default="v1")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ModelMetrics(Base):
    __tablename__ = "model_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    epoch: Mapped[int] = mapped_column(Integer, nullable=False)
    map_50: Mapped[float] = mapped_column(Float, nullable=True)
    map_75: Mapped[float] = mapped_column(Float, nullable=True)
    map_overall: Mapped[float] = mapped_column(Float, nullable=True)
    loss_cls: Mapped[float] = mapped_column(Float, nullable=True)
    loss_bbox: Mapped[float] = mapped_column(Float, nullable=True)
    loss_total: Mapped[float] = mapped_column(Float, nullable=True)
    notes: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
