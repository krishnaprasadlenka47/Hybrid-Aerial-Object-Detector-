from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str
    checkpoint_dir: str = "data/checkpoints"
    image_dir: str = "data/images"
    results_dir: str = "data/results"

    image_size: int = 1024
    num_classes: int = 15
    confidence_threshold: float = 0.5
    nms_iou_threshold: float = 0.4

    batch_size: int = 4
    epochs: int = 12
    learning_rate: float = 0.0001

    app_env: str = "development"

    class Config:
        env_file = ".env"


settings = Settings()

CATEGORY_NAMES = [
    "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
    "basketball-court", "ground-track-field", "harbor", "bridge", "large-vehicle",
    "small-vehicle", "helicopter", "roundabout", "soccer-ball-field", "swimming-pool"
]
