from PIL import Image
import requests

from app.clients.nvidia_vlm_client import (
    INITIAL_SYSTEM_PROMPT,
    POSE_AUDIT_SYSTEM_PROMPT,
    POSE_AUDIT_USER_PROMPT,
    REFINEMENT_SYSTEM_PROMPT,
    _apply_pose_audit,
    _build_refinement_instruction,
    _parse_json_object,
    _parse_prompt_result,
)
from app.core.image_io import image_to_png_data_url
from app.models.schemas import PromptExtractionResult, SimilarityScore


class OllamaVisionClient:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate_initial_prompt(self, image_data_url: str) -> PromptExtractionResult:
        content = self._chat(
            system_prompt=INITIAL_SYSTEM_PROMPT,
            user_prompt=(
                "Create a detailed image-generation prompt for this image. If people are visible, "
                "start with exact foreground-person position, pose, gaze, body direction, walking "
                "trajectory, hand placement, interaction, hair, clothing, and expression. Return JSON only."
            ),
            image_data_urls=[image_data_url],
            temperature=0.2,
        )
        prompt_result = _parse_prompt_result(content)
        pose_audit = self._classify_pose_and_motion(image_data_url)
        return _apply_pose_audit(prompt_result, pose_audit)

    def refine_prompt(
        self,
        original_image_data_url: str,
        generated_image: Image.Image,
        previous_prompt: str,
        previous_negative_prompt: str,
        similarity_report: SimilarityScore,
    ) -> PromptExtractionResult:
        content = self._chat(
            system_prompt=REFINEMENT_SYSTEM_PROMPT,
            user_prompt=_build_refinement_instruction(
                previous_prompt=previous_prompt,
                previous_negative_prompt=previous_negative_prompt,
                similarity_report=similarity_report,
            ),
            image_data_urls=[original_image_data_url, image_to_png_data_url(generated_image)],
            temperature=0.25,
        )
        return _parse_prompt_result(content)

    def _classify_pose_and_motion(self, image_data_url: str) -> dict:
        content = self._chat(
            system_prompt=POSE_AUDIT_SYSTEM_PROMPT,
            user_prompt=POSE_AUDIT_USER_PROMPT,
            image_data_urls=[image_data_url],
            temperature=0,
        )
        return _parse_json_object(content)

    def _chat(self, system_prompt: str, user_prompt: str, image_data_urls: list[str], temperature: float) -> str:
        if not self.model.strip():
            raise RuntimeError("Ollama VLM model is required.")

        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "options": {"temperature": temperature},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": user_prompt,
                        "images": [_data_url_to_base64(data_url) for data_url in image_data_urls],
                    },
                ],
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]


def _data_url_to_base64(data_url: str) -> str:
    if "," not in data_url:
        return data_url
    return data_url.split(",", 1)[1]
