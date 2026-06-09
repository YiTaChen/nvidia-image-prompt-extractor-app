from __future__ import annotations

from app.clients.nvidia_image_client import NvidiaImageGenerationClient
from app.core.config import Settings, get_settings
from app.models.schemas import ImageGenerationRequest


CANDIDATE_MODELS = [
    "stabilityai/sdxl-turbo",
    "stabilityai/stable-diffusion-3-medium",
    "stabilityai/stable-diffusion-xl",
]


def main() -> None:
    base_settings = get_settings()
    request = ImageGenerationRequest(
        prompt="A small green plant on a white desk, soft natural light",
        width=512,
        height=512,
        seed=0,
    )

    for model in CANDIDATE_MODELS:
        settings = Settings(
            nvidia_api_key=base_settings.nvidia_api_key,
            nvidia_image_model=model,
            use_mock_nvidia=False,
        )
        try:
            result = NvidiaImageGenerationClient(settings).generate_image(request)
        except Exception as exc:
            print(f"FAIL {model}: {exc}")
            continue
        print(f"OK {model}: {result.mime_type}, {len(result.image_base64)} base64 chars")
        return

    raise SystemExit("No hosted NVIDIA image-generation endpoint is available for this API key.")


if __name__ == "__main__":
    main()
