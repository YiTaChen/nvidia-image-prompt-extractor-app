import base64

import requests

from app.core.config import Settings
from app.models.schemas import ImageGenerationRequest, ImageGenerationResult


class NvidiaImageGenerationClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        if not self.settings.has_nvidia_key:
            raise RuntimeError("NVIDIA_API_KEY is required for real NVIDIA calls.")

        if _is_legacy_stability_model(self.settings.nvidia_image_model):
            return self._generate_legacy_stability_image(request)

        payload = {
            "model": self.settings.nvidia_image_model,
            "prompt": request.prompt,
            "n": 1,
            "response_format": "b64_json",
            "size": f"{request.width}x{request.height}",
        }
        if request.negative_prompt:
            payload["negative_prompt"] = request.negative_prompt
        if request.seed is not None:
            payload["seed"] = request.seed

        response = requests.post(
            f"{self.settings.resolved_image_base_url}/images/generations",
            headers={
                "Authorization": f"Bearer {self.settings.nvidia_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=180,
        )
        response.raise_for_status()
        data = response.json()["data"][0]
        image_base64 = data.get("b64_json")
        if not image_base64 and data.get("url"):
            image_response = requests.get(data["url"], timeout=90)
            image_response.raise_for_status()
            image_base64 = base64.b64encode(image_response.content).decode("ascii")
        if not image_base64:
            raise RuntimeError("NVIDIA image generation response did not include image data.")
        return ImageGenerationResult(
            image_base64=image_base64,
            mime_type="image/png",
            model=self.settings.nvidia_image_model,
        )

    def _generate_legacy_stability_image(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        response = requests.post(
            f"https://ai.api.nvidia.com/v1/genai/{self.settings.nvidia_image_model}",
            headers={
                "Authorization": f"Bearer {self.settings.nvidia_api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            json=_build_legacy_stability_payload(request),
            timeout=180,
        )
        response.raise_for_status()
        data = response.json()
        artifact = data["artifacts"][0]
        return ImageGenerationResult(
            image_base64=artifact["base64"],
            mime_type="image/jpeg",
            model=self.settings.nvidia_image_model,
        )


def _is_legacy_stability_model(model: str) -> bool:
    return model in {
        "stabilityai/stable-diffusion-xl",
        "stabilityai/stable-diffusion-3-medium",
    }


def _build_legacy_stability_payload(request: ImageGenerationRequest) -> dict:
    if request.width != 1024 or request.height != 1024:
        width = 1024
        height = 1024
    else:
        width = request.width
        height = request.height
    payload = {
        "height": height,
        "width": width,
        "text_prompts": [{"text": request.prompt, "weight": 1}],
    }
    if request.negative_prompt:
        payload["text_prompts"].append({"text": request.negative_prompt, "weight": -1})
    if request.seed is not None:
        payload["seed"] = request.seed
    return payload
