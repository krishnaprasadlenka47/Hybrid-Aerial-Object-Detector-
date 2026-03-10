from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.db.database import init_db
from app.routers import detection


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="Hybrid Aerial Object Detector",
    description="Strip R-CNN with Vision Transformer contextual encoding for UAV imagery detection across 15 object categories.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(detection.router)


@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "message": "Aerial Detection API is running"}
