from app.clients.mock_clients import MockImageGenerationClient, MockVisionClient
from app.clients.nvidia_image_client import _build_legacy_stability_payload, _is_legacy_stability_model
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
    assert _is_legacy_stability_model("stabilityai/stable-diffusion-xl") is True
    assert _is_legacy_stability_model("qwen/qwen-image") is False


def test_legacy_stability_payload_includes_negative_prompt():
    payload = _build_legacy_stability_payload(
        ImageGenerationRequest(prompt="plant", negative_prompt="blur", width=1024, height=1024)
    )

    assert payload["text_prompts"][0] == {"text": "plant", "weight": 1}
    assert payload["text_prompts"][1] == {"text": "blur", "weight": -1}
