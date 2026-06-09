from PIL import Image

from app.clients.nvidia_vlm_client import (
    INITIAL_SYSTEM_PROMPT,
    NvidiaVisionClient,
    _apply_pose_audit,
    _build_refinement_instruction,
)
from app.core.config import Settings
from app.models.schemas import PromptExtractionResult, SimilarityScore


class FakeResponse:
    def __init__(self, content='{"prompt":"refined prompt","negative_prompt":"blur","analysis":{"summary":"ok"}}'):
        self.content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "choices": [
                {
                    "message": {
                        "content": self.content
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


def test_prompts_prioritize_visible_human_identity_clothing_and_pose():
    instruction = _build_refinement_instruction(
        previous_prompt="old prompt",
        previous_negative_prompt="old negative",
        similarity_report=SimilarityScore(final_score=42, histogram_score=50, average_hash_score=30),
    )

    combined = f"{INITIAL_SYSTEM_PROMPT}\n{instruction}".lower()

    assert "visible skin tone" in combined
    assert "hair color" in combined
    assert "clothing" in combined
    assert "pose" in combined
    assert "foreground people" in combined
    assert "prompt text itself" in combined
    assert "left hand" in combined
    assert "right hand" in combined
    assert "gaze direction" in combined
    assert "leg action" in combined
    assert "hand contact" in combined
    assert "walking trajectory" in combined
    assert "movement direction" in combined
    assert "head direction" in combined
    assert "body direction" in combined
    assert "screen-space" in combined


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


def test_generate_initial_prompt_applies_pose_audit(monkeypatch):
    responses = [
        FakeResponse(
            '{"prompt":"A couple walking towards the camera on a sidewalk.",'
            '"negative_prompt":"blur","analysis":{"human_subjects":[]}}'
        ),
        FakeResponse(
            '{"faces_gaze":"at camera","body_direction":"diagonal_screen_right_away",'
            '"walking_trajectory":"rightward and away from camera",'
            '"corrected_action_sentence":"The couple is walking away from the camera, diagonally to the right."}'
        ),
    ]

    def fake_post(url, headers, json, timeout):
        return responses.pop(0)

    monkeypatch.setattr("app.clients.nvidia_vlm_client.requests.post", fake_post)

    client = NvidiaVisionClient(Settings(_env_file=None, nvidia_api_key="key"))
    result = client.generate_initial_prompt("data:image/png;base64,original")

    assert result.prompt.startswith("Global action direction to match exactly:")
    assert "diagonal_screen_right_away" in result.prompt
    assert "walking towards the camera" not in result.prompt


def test_pose_audit_removes_structured_toward_camera_contradictions():
    result = PromptExtractionResult(
        prompt=(
            "Foreground people details to match exactly: person 1: head direction: toward camera; "
            "body direction: toward camera; movement direction: toward camera; "
            "walking trajectory: diagonal path; camera relation: close to camera. "
            "They are walking towards the camera."
        ),
        negative_prompt="blur",
        analysis={},
    )

    updated = _apply_pose_audit(
        result,
        {
            "faces_gaze": "toward_camera",
            "body_direction": "diagonal_screen_right_away",
            "walking_trajectory": "rightward and away from camera along the sidewalk",
            "camera_relation": "faces look back while bodies move away",
            "corrected_action_sentence": "The couple's faces look back at the camera while their bodies walk diagonally toward the right side of the frame, away along the sidewalk.",
        },
    )

    assert "body direction: toward camera" not in updated.prompt
    assert "movement direction: toward camera" not in updated.prompt
    assert "walking towards the camera" not in updated.prompt
    assert "body direction: diagonal_screen_right_away" in updated.prompt
    assert "walking trajectory: diagonally toward viewer-right and away from camera along the sidewalk" in updated.prompt
    assert "camera relation: faces and gaze look back toward camera while bodies move away" in updated.prompt
