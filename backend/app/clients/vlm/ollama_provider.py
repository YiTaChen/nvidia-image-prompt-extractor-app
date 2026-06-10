import requests

from app.clients.vlm.base import VlmProvider, merge_discovered_with_reference, reference_model_infos
from app.core.config import Settings
from app.models.schemas import VlmModelListResponse


class OllamaVlmProvider(VlmProvider):
    provider_id = "ollama"
    display_name = "Ollama"
    reference_models = (
        "llava:latest",
        "llama3.2-vision:11b",
        "minicpm-v:latest",
        "moondream:latest",
        "bakllava:latest",
    )

    def default_base_url(self, settings: Settings) -> str:
        return settings.ollama_base_url

    def default_model(self, settings: Settings) -> str:
        if settings.ollama_vlm_model:
            return settings.ollama_vlm_model
        if settings.vlm_provider == self.provider_id and settings.vlm_model:
            return settings.vlm_model
        return self.reference_models[0]

    def list_models(
        self,
        settings: Settings,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> VlmModelListResponse:
        connection = self.resolve_connection(settings, base_url, api_key)
        try:
            response = requests.get(f"{connection.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            model_ids = _ollama_model_ids(response.json())
        except Exception as exc:
            return VlmModelListResponse(
                provider=self.provider_id,
                connection_status="failed",
                message=f"Could not query Ollama; showing reference models. {exc}",
                models=reference_model_infos(
                    self.provider_id,
                    self.reference_models,
                    "Start Ollama and pull a vision-capable model to make it available.",
                ),
            )

        return VlmModelListResponse(
            provider=self.provider_id,
            connection_status="ok",
            message="Ollama model discovery succeeded.",
            models=merge_discovered_with_reference(self.provider_id, model_ids, self.reference_models),
        )


def _ollama_model_ids(payload: dict) -> list[str]:
    models = payload.get("models")
    if not isinstance(models, list):
        return []
    return [item.get("name", "") for item in models if isinstance(item, dict)]
