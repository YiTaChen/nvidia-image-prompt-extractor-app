import requests

from app.clients.vlm.base import VlmProvider, merge_discovered_with_reference, reference_model_infos
from app.core.config import Settings
from app.models.schemas import VlmModelListResponse


class LmStudioVlmProvider(VlmProvider):
    provider_id = "lm_studio"
    display_name = "LM Studio"
    reference_models = (
        "llava-v1.6",
        "llama-3.2-vision",
        "qwen2.5-vl-7b-instruct",
        "minicpm-v-2.6",
    )

    def default_base_url(self, settings: Settings) -> str:
        return settings.lm_studio_base_url

    def default_api_key(self, settings: Settings) -> str:
        return settings.lm_studio_api_key

    def default_model(self, settings: Settings) -> str:
        if settings.lm_studio_vlm_model:
            return settings.lm_studio_vlm_model
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
            headers = {}
            if connection.api_key.strip():
                headers["Authorization"] = f"Bearer {connection.api_key}"
            response = requests.get(f"{connection.base_url}/models", headers=headers, timeout=10)
            response.raise_for_status()
            model_ids = _openai_model_ids(response.json())
        except Exception as exc:
            return VlmModelListResponse(
                provider=self.provider_id,
                connection_status="failed",
                message=f"Could not query LM Studio; showing reference models. {exc}",
                models=reference_model_infos(
                    self.provider_id,
                    self.reference_models,
                    "Start LM Studio's local server and load a vision-capable model to make it available.",
                ),
            )

        return VlmModelListResponse(
            provider=self.provider_id,
            connection_status="ok",
            message="LM Studio model discovery succeeded.",
            models=merge_discovered_with_reference(self.provider_id, model_ids, self.reference_models),
        )


def _openai_model_ids(payload: dict) -> list[str]:
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    return [item.get("id", "") for item in data if isinstance(item, dict)]
