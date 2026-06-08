from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app


FIXTURE = Path(__file__).parent / "fixtures" / "sample.png"


def test_extract_prompt_endpoint_with_mock_client():
    get_settings.cache_clear()
    settings = get_settings()
    settings.use_mock_nvidia = True
    client = TestClient(app)

    with FIXTURE.open("rb") as image:
        response = client.post("/api/extract-prompt", files={"image": ("sample.png", image, "image/png")})

    assert response.status_code == 200
    assert response.json()["prompt"]


def test_generate_image_endpoint_with_mock_client():
    get_settings.cache_clear()
    settings = get_settings()
    settings.use_mock_nvidia = True
    client = TestClient(app)

    response = client.post("/api/generate-image", json={"prompt": "a quiet desk", "width": 256, "height": 256})

    assert response.status_code == 200
    assert response.json()["image_base64"]
