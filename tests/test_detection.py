import io
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from PIL import Image
from app.main import app

client = TestClient(app)


def make_test_image() -> bytes:
    img = Image.new("RGB", (256, 256), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_health():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_detect_unsupported_type():
    response = client.post(
        "/detect/",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415


def test_detect_missing_checkpoint():
    image_bytes = make_test_image()
    response = client.post(
        "/detect/",
        files={"file": ("uav.jpg", image_bytes, "image/jpeg")},
    )
    assert response.status_code in [503, 200]
