import base64
from urllib.parse import quote

import requests

from app.core.config import Settings
from app.models.schemas import ImageGenerationRequest, ImageGenerationResult


class PollinationsImageGenerationClient:
    def __init__(self, settings: Settings, api_key: str | None = None):
        self.settings = settings
        self.api_key = api_key if api_key is not None else settings.pollinations_api_key

    def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        if not self.api_key.strip():
            raise RuntimeError("POLLINATIONS_API_KEY is required for Pollinations image generation.")

        model = request.image_model or self.settings.pollinations_model
        url = _build_pollinations_url(request.prompt, model, request.width, request.height)
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=180,
        )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "image/jpeg").split(";")[0]
        if not content_type.startswith("image/"):
            raise RuntimeError(f"Pollinations did not return an image. Content-Type: {content_type}")
        return ImageGenerationResult(
            image_base64=base64.b64encode(response.content).decode("ascii"),
            mime_type=content_type,
            model=f"pollinations/{model}",
            provider="pollinations",
        )


def _build_pollinations_url(prompt: str, model: str, width: int, height: int) -> str:
    encoded_prompt = quote(prompt)
    return (
        f"https://gen.pollinations.ai/image/{encoded_prompt}"
        f"?model={quote(model)}&width={width}&height={height}&nologo=true&private=true"
    )
