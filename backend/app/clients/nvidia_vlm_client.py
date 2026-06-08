import json

import requests

from app.core.config import Settings
from app.models.schemas import PromptExtractionResult


INITIAL_SYSTEM_PROMPT = (
    "You are an expert image prompt engineer. Analyze the uploaded image and produce "
    "a precise prompt for a text-to-image generation model. Preserve subject, "
    "composition, style, lighting, color palette, camera details, and important "
    "visual details. Return strict JSON only with prompt, negative_prompt, and analysis."
)


class NvidiaVisionClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def generate_initial_prompt(self, image_data_url: str) -> PromptExtractionResult:
        if not self.settings.has_nvidia_key:
            raise RuntimeError("NVIDIA_API_KEY is required for real NVIDIA calls.")

        response = requests.post(
            f"{self.settings.nvidia_base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.settings.nvidia_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.settings.nvidia_vlm_model,
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": INITIAL_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Create a detailed image-generation prompt for this image. Return JSON only.",
                            },
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                        ],
                    },
                ],
            },
            timeout=90,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return _parse_prompt_result(content)


def _parse_prompt_result(content: str) -> PromptExtractionResult:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.removeprefix("json").strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        payload = {"prompt": cleaned, "negative_prompt": "", "analysis": {}}
    if isinstance(payload.get("analysis"), str):
        payload["analysis"] = {"summary": payload["analysis"]}
    if not isinstance(payload.get("analysis", {}), dict):
        payload["analysis"] = {}
    return PromptExtractionResult.model_validate(payload)
