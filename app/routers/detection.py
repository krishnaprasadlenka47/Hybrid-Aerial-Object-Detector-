from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.schemas.detection import DetectionResponse, StatsResponse
from app.services.detector_service import run_detection, get_stats

router = APIRouter(prefix="/detect", tags=["Detection"])

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/tiff", "image/webp"}


@router.post("/", response_model=DetectionResponse)
async def detect(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}.",
        )

    image_bytes = await file.read()
    if len(image_bytes) > 50 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image exceeds 50 MB limit.",
        )

    return await run_detection(image_bytes, file.filename, db)


@router.get("/stats", response_model=StatsResponse)
async def stats(
    limit: int = 10,
    db: AsyncSession = Depends(get_db),
):
    return await get_stats(db, limit)
