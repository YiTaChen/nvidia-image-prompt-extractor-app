import requests
from PIL import Image

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


class OpenAiCompatibleVisionClient:
    def __init__(self, base_url: str, model: str, api_key: str = "", provider_name: str = "OpenAI-compatible"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.provider_name = provider_name

    def generate_initial_prompt(self, image_data_url: str) -> PromptExtractionResult:
        content = self._chat(
            temperature=0.2,
            messages=[
                {"role": "system", "content": INITIAL_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Create a detailed image-generation prompt for this image. "
                                "If people are visible, start with exact foreground-person "
                                "position, pose, gaze, body direction, walking trajectory, "
                                "hand placement, interaction, hair, clothing, and expression. "
                                "Return JSON only."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                },
            ],
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
            temperature=0.25,
            messages=[
                {"role": "system", "content": REFINEMENT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": _build_refinement_instruction(
                                previous_prompt=previous_prompt,
                                previous_negative_prompt=previous_negative_prompt,
                                similarity_report=similarity_report,
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": original_image_data_url}},
                        {"type": "image_url", "image_url": {"url": image_to_png_data_url(generated_image)}},
                    ],
                },
            ],
        )
        return _parse_prompt_result(content)

    def _classify_pose_and_motion(self, image_data_url: str) -> dict:
        content = self._chat(
            temperature=0,
            messages=[
                {"role": "system", "content": POSE_AUDIT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": POSE_AUDIT_USER_PROMPT},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                },
            ],
        )
        return _parse_json_object(content)

    def _chat(self, messages: list[dict], temperature: float) -> str:
        if not self.model.strip():
            raise RuntimeError(f"{self.provider_name} VLM model is required.")

        headers = {"Content-Type": "application/json"}
        if self.api_key.strip():
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "temperature": temperature,
                "messages": messages,
            },
            timeout=90,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
