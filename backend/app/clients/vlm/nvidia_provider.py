import requests

from app.clients.vlm.base import VlmProvider, merge_discovered_with_reference, reference_model_infos
from app.core.config import Settings
from app.models.schemas import VlmModelListResponse


class NvidiaVlmProvider(VlmProvider):
    provider_id = "nvidia"
    display_name = "NVIDIA"
    requires_api_key = True
    reference_models = (
        "nvidia/nemotron-nano-12b-v2-vl",
        "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
        "meta/llama-3.2-11b-vision-instruct",
        "meta/llama-3.2-90b-vision-instruct",
    )

    def default_base_url(self, settings: Settings) -> str:
        return settings.nvidia_base_url

    def default_api_key(self, settings: Settings) -> str:
        return settings.nvidia_api_key

    def default_model(self, settings: Settings) -> str:
        return settings.nvidia_vlm_model or settings.vlm_model or self.reference_models[0]

    def list_models(
        self,
        settings: Settings,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> VlmModelListResponse:
        connection = self.resolve_connection(settings, base_url, api_key)
        if not connection.api_key.strip():
            return VlmModelListResponse(
                provider=self.provider_id,
                connection_status="missing_key",
                message="NVIDIA API key is not configured; showing reference models.",
                models=reference_model_infos(
                    self.provider_id,
                    self.reference_models,
                    "NVIDIA API key is required to discover currently available hosted models.",
                ),
            )

        try:
            response = requests.get(
                f"{connection.base_url}/models",
                headers={"Authorization": f"Bearer {connection.api_key}"},
                timeout=30,
            )
            response.raise_for_status()
            model_ids = _openai_model_ids(response.json())
        except Exception as exc:
            return VlmModelListResponse(
                provider=self.provider_id,
                connection_status="failed",
                message=f"Could not query NVIDIA models; showing reference models. {exc}",
                models=reference_model_infos(
                    self.provider_id,
                    self.reference_models,
                    "Model discovery failed; verify the NVIDIA API key and base URL.",
                ),
            )

        return VlmModelListResponse(
            provider=self.provider_id,
            connection_status="ok",
            message="NVIDIA model discovery succeeded.",
            models=merge_discovered_with_reference(self.provider_id, model_ids, self.reference_models),
        )


def _openai_model_ids(payload: dict) -> list[str]:
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    return [item.get("id", "") for item in data if isinstance(item, dict)]
