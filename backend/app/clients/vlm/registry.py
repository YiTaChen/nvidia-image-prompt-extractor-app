from app.clients.vlm.base import VlmProvider
from app.clients.vlm.lm_studio_provider import LmStudioVlmProvider
from app.clients.vlm.nvidia_provider import NvidiaVlmProvider
from app.clients.vlm.ollama_provider import OllamaVlmProvider


_PROVIDERS: dict[str, VlmProvider] = {
    provider.provider_id: provider
    for provider in (
        NvidiaVlmProvider(),
        LmStudioVlmProvider(),
        OllamaVlmProvider(),
    )
}


def list_vlm_providers() -> list[VlmProvider]:
    return list(_PROVIDERS.values())


def get_vlm_provider(provider_id: str) -> VlmProvider:
    normalized = provider_id.strip().lower()
    if normalized not in _PROVIDERS:
        raise KeyError(provider_id)
    return _PROVIDERS[normalized]
