from PIL import Image

from app.clients.nvidia_vlm_client import NvidiaVisionClient, _build_refinement_instruction
from app.core.config import Settings
from app.models.schemas import SimilarityScore


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"prompt":"refined prompt","negative_prompt":"blur","analysis":{"summary":"ok"}}'
                    }
                }
            ]
        }


def test_refinement_instruction_includes_prompt_and_scores():
    instruction = _build_refinement_instruction(
        previous_prompt="old prompt",
        previous_negative_prompt="old negative",
        similarity_report=SimilarityScore(final_score=42, histogram_score=50, average_hash_score=30),
    )

    assert "old prompt" in instruction
    assert "old negative" in instruction
    assert "final_score: 42.0" in instruction


def test_refine_prompt_sends_original_generated_image_and_report(monkeypatch):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("app.clients.nvidia_vlm_client.requests.post", fake_post)

    client = NvidiaVisionClient(Settings(_env_file=None, nvidia_api_key="key"))
    result = client.refine_prompt(
        original_image_data_url="data:image/png;base64,original",
        generated_image=Image.new("RGB", (8, 8), (12, 34, 56)),
        previous_prompt="old prompt",
        previous_negative_prompt="old negative",
        similarity_report=SimilarityScore(final_score=42, histogram_score=50, average_hash_score=30),
    )

    content = captured["json"]["messages"][1]["content"]
    image_parts = [part for part in content if part["type"] == "image_url"]

    assert result.prompt == "refined prompt"
    assert captured["url"].endswith("/chat/completions")
    assert len(image_parts) == 2
    assert image_parts[0]["image_url"]["url"] == "data:image/png;base64,original"
    assert image_parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "old prompt" in content[0]["text"]
