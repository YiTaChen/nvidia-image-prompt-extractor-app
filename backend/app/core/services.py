from app.clients.mock_clients import MockImageGenerationClient, MockVisionClient
from app.clients.nvidia_image_client import NvidiaImageGenerationClient
from app.clients.nvidia_vlm_client import NvidiaVisionClient
from app.clients.ollama_vlm_client import OllamaVisionClient
from app.clients.openai_compatible_vlm_client import OpenAiCompatibleVisionClient
from app.clients.pollinations_image_client import PollinationsImageGenerationClient
from app.core.config import Settings


def get_vision_client(
    settings: Settings,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
):
    if settings.use_mock_nvidia:
        return MockVisionClient()

    selected_provider = (provider or settings.vlm_provider or "nvidia").strip().lower()
    if selected_provider == "nvidia":
        return NvidiaVisionClient(
            settings,
            api_key=api_key,
            base_url=base_url,
            model=model or settings.nvidia_vlm_model or _generic_model(settings, "nvidia"),
        )
    if selected_provider == "lm_studio":
        return OpenAiCompatibleVisionClient(
            base_url=base_url or settings.lm_studio_base_url,
            model=model or settings.lm_studio_vlm_model or _generic_model(settings, "lm_studio"),
            api_key=api_key if api_key is not None else settings.lm_studio_api_key,
            provider_name="LM Studio",
        )
    if selected_provider == "ollama":
        return OllamaVisionClient(
            base_url=base_url or settings.ollama_base_url,
            model=model or settings.ollama_vlm_model or _generic_model(settings, "ollama"),
        )
    raise ValueError(f"Unsupported VLM provider: {selected_provider}")


def _generic_model(settings: Settings, provider: str) -> str:
    if (settings.vlm_provider or "").strip().lower() != provider:
        return ""
    return settings.vlm_model


def get_image_generation_client(settings: Settings):
    if settings.use_mock_nvidia:
        return MockImageGenerationClient()
    if settings.image_provider == "pollinations":
        return PollinationsImageGenerationClient(settings)
    return NvidiaImageGenerationClient(settings)
