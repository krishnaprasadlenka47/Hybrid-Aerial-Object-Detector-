import os
import time
import torch
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from fastapi import HTTPException, status

from app.ml.backbone.vit_encoder import ViTEncoder
from app.ml.detector.strip_rcnn import HybridAerialDetector
from app.ml.utils.transforms import load_image_tensor
from app.ml.utils.postprocess import generate_anchors, postprocess_detections
from app.models.detection import DetectionLog
from app.schemas.detection import DetectionResponse, DetectionItem, BoundingBox
from app.config import settings, CATEGORY_NAMES

_vit: ViTEncoder | None = None
_detector: HybridAerialDetector | None = None


def load_models():
    global _vit, _detector

    if _vit is not None and _detector is not None:
        return _vit, _detector

    vit_ckpt = os.path.join(settings.checkpoint_dir, "vit_encoder.pt")
    det_ckpt = os.path.join(settings.checkpoint_dir, "detector.pt")

    if not os.path.exists(vit_ckpt) or not os.path.exists(det_ckpt):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model checkpoints not found. Run training first: python -m app.ml.train",
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vit = ViTEncoder(img_size=settings.image_size, patch_size=16, embed_dim=768, depth=6, num_heads=12)
    vit.load_state_dict(torch.load(vit_ckpt, map_location=device))
    vit.eval()

    detector = HybridAerialDetector(num_classes=settings.num_classes, vit_embed_dim=768, base_channels=64)
    detector.load_state_dict(torch.load(det_ckpt, map_location=device))
    detector.eval()

    _vit = vit
    _detector = detector
    return _vit, _detector


async def run_detection(image_bytes: bytes, filename: str, db: AsyncSession) -> DetectionResponse:
    vit, detector = load_models()
    device = next(vit.parameters()).device

    tensor, original_size = load_image_tensor(image_bytes, settings.image_size)
    tensor = tensor.to(device)

    start = time.time()
    with torch.no_grad():
        vit_features = vit(tensor)
        outputs = detector(tensor, vit_features)

    inference_ms = round((time.time() - start) * 1000, 2)

    feat_h = outputs["fpn_p3"].shape[2]
    feat_w = outputs["fpn_p3"].shape[3]
    anchors = generate_anchors(feat_h, feat_w, stride=8).to(device)

    raw_results = postprocess_detections(
        outputs["cls_logits"],
        outputs["bbox_deltas"],
        anchors,
        settings.image_size,
    )

    detection_items = []
    for r in raw_results:
        cat_id = r["category_id"]
        cat_name = CATEGORY_NAMES[cat_id - 1] if 1 <= cat_id <= len(CATEGORY_NAMES) else "unknown"
        detection_items.append(DetectionItem(
            category_id=cat_id,
            category_name=cat_name,
            confidence=r["confidence"],
            bbox=BoundingBox(**r["bbox"]),
        ))

    log = DetectionLog(
        filename=filename,
        image_width=original_size[0],
        image_height=original_size[1],
        total_detections=len(detection_items),
        detections=[d.model_dump() for d in detection_items],
        inference_time_ms=inference_ms,
        model_version="v1",
    )
    db.add(log)
    await db.commit()
    await db.refresh(log)

    return DetectionResponse(
        filename=filename,
        image_width=original_size[0],
        image_height=original_size[1],
        total_detections=len(detection_items),
        detections=detection_items,
        inference_time_ms=inference_ms,
        model_version="v1",
    )


async def get_stats(db: AsyncSession, limit: int = 10) -> dict:
    total_images = await db.scalar(select(func.count()).select_from(DetectionLog))
    total_objects = await db.scalar(select(func.sum(DetectionLog.total_detections)).select_from(DetectionLog)) or 0
    avg_detections = round(total_objects / total_images, 2) if total_images > 0 else 0.0

    avg_time_result = await db.scalar(select(func.avg(DetectionLog.inference_time_ms)).select_from(DetectionLog))
    avg_time = round(avg_time_result or 0.0, 2)

    recent_result = await db.execute(
        select(DetectionLog).order_by(DetectionLog.created_at.desc()).limit(limit)
    )
    recent_logs = recent_result.scalars().all()

    category_counts: dict[str, int] = {}
    for log in recent_logs:
        for det in log.detections:
            name = det.get("category_name", "unknown")
            category_counts[name] = category_counts.get(name, 0) + 1

    return {
        "total_images_processed": total_images,
        "total_objects_detected": total_objects,
        "avg_detections_per_image": avg_detections,
        "avg_inference_time_ms": avg_time,
        "category_counts": category_counts,
        "recent_logs": recent_logs,
    }
