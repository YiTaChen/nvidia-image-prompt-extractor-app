import base64

import requests

from app.core.config import Settings
from app.models.schemas import ImageGenerationRequest, ImageGenerationResult


HOSTED_NVIDIA_LLM_BASE_URL = "integrate.api.nvidia.com"


class NvidiaImageGenerationClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        if not self.settings.has_nvidia_key:
            raise RuntimeError("NVIDIA_API_KEY is required for real NVIDIA calls.")

        if _is_legacy_stability_model(self.settings.nvidia_image_model):
            return self._generate_legacy_stability_image(request)

        _validate_openai_compatible_image_endpoint(self.settings)

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
            _openai_compatible_image_generation_url(self.settings),
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
        "stabilityai/sdxl-turbo",
        "stabilityai/stable-diffusion-xl",
        "stabilityai/stable-diffusion-3-medium",
    }


def _validate_openai_compatible_image_endpoint(settings: Settings) -> None:
    if settings.has_dedicated_image_base_url:
        return
    if HOSTED_NVIDIA_LLM_BASE_URL in settings.nvidia_base_url:
        raise RuntimeError(
            "NVIDIA_IMAGE_BASE_URL is required for NVIDIA image generation. "
            "The hosted NVIDIA_BASE_URL is for VLM/chat models and does not expose "
            "/images/generations for Visual GenAI NIMs. Start a Visual GenAI NIM "
            "such as Qwen-Image, then set NVIDIA_IMAGE_BASE_URL to its /v1 base URL, "
            "for example http://localhost:8000/v1."
        )


def _openai_compatible_image_generation_url(settings: Settings) -> str:
    base_url = settings.resolved_image_base_url
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return f"{base_url}/images/generations"


def _build_legacy_stability_payload(request: ImageGenerationRequest) -> dict:
    if request.width != 1024 or request.height != 1024:
        width = 1024
        height = 1024
    else:
        width = request.width
        height = request.height
    payload = {
        "text_prompts": [{"text": request.prompt, "weight": 1}],
        "seed": request.seed or 0,
    }
    if request.width == 1024 and request.height == 1024:
        payload["height"] = height
        payload["width"] = width
    if request.prompt and request.width != 1024:
        payload["sampler"] = "K_EULER_ANCESTRAL"
        payload["steps"] = 2
    if request.negative_prompt:
        payload["text_prompts"].append({"text": request.negative_prompt, "weight": -1})
    return payload
