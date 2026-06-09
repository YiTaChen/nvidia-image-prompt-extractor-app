from app.clients.mock_clients import MockImageGenerationClient, MockVisionClient
import pytest

from app.clients.nvidia_image_client import (
    _build_legacy_stability_payload,
    _openai_compatible_image_generation_url,
    _validate_openai_compatible_image_endpoint,
    _is_legacy_stability_model,
)
from app.clients.pollinations_image_client import _build_pollinations_url
from app.core.config import Settings
from app.models.schemas import ImageGenerationRequest


def test_mock_vision_client_returns_structured_prompt():
    result = MockVisionClient().generate_initial_prompt("data:image/png;base64,abc")

    assert result.prompt
    assert "composition" in result.analysis


def test_mock_image_client_returns_base64_image():
    result = MockImageGenerationClient().generate_image(ImageGenerationRequest(prompt="test prompt", width=256, height=256))

    assert result.image_base64
    assert result.mime_type == "image/png"


def test_legacy_stability_model_detection():
    assert _is_legacy_stability_model("stabilityai/sdxl-turbo") is True
    assert _is_legacy_stability_model("stabilityai/stable-diffusion-xl") is True
    assert _is_legacy_stability_model("qwen/qwen-image") is False


def test_legacy_stability_payload_includes_negative_prompt():
    payload = _build_legacy_stability_payload(
        ImageGenerationRequest(prompt="plant", negative_prompt="blur", width=1024, height=1024)
    )

    assert payload["text_prompts"][0] == {"text": "plant", "weight": 1}
    assert payload["text_prompts"][1] == {"text": "blur", "weight": -1}
    assert payload["seed"] == 0


def test_openai_compatible_image_endpoint_requires_dedicated_base_url_for_hosted_nvidia():
    settings = Settings(
        nvidia_api_key="key",
        nvidia_base_url="https://integrate.api.nvidia.com/v1",
        nvidia_image_base_url="",
    )

    with pytest.raises(RuntimeError, match="NVIDIA_IMAGE_BASE_URL is required"):
        _validate_openai_compatible_image_endpoint(settings)


def test_openai_compatible_image_generation_url_adds_v1_when_needed():
    settings = Settings(
        nvidia_api_key="key",
        nvidia_image_base_url="http://localhost:8000",
    )

    assert _openai_compatible_image_generation_url(settings) == "http://localhost:8000/v1/images/generations"


def test_openai_compatible_image_generation_url_keeps_v1_base():
    settings = Settings(
        nvidia_api_key="key",
        nvidia_image_base_url="http://localhost:8000/v1",
    )

    assert _openai_compatible_image_generation_url(settings) == "http://localhost:8000/v1/images/generations"


def test_pollinations_url_includes_model_size_and_private_flags():
    url = _build_pollinations_url("green plant", "kontext", 512, 512)

    assert url.startswith("https://gen.pollinations.ai/image/green%20plant")
    assert "model=kontext" in url
    assert "width=512" in url
    assert "height=512" in url
    assert "private=true" in url
