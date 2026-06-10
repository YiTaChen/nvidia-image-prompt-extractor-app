from dataclasses import dataclass

from app.core.config import Settings
from app.models.schemas import VlmModelInfo, VlmProviderInfo


@dataclass(frozen=True)
class VlmConnection:
    provider: str
    base_url: str
    api_key: str


class VlmProvider:
    provider_id: str
    display_name: str
    requires_api_key: bool = False
    supports_custom_base_url: bool = True
    reference_models: tuple[str, ...] = ()

    def provider_info(self, settings: Settings) -> VlmProviderInfo:
        return VlmProviderInfo(
            id=self.provider_id,
            display_name=self.display_name,
            default_base_url=self.default_base_url(settings),
            default_model=self.default_model(settings),
            requires_api_key=self.requires_api_key,
            api_key_configured=bool(self.default_api_key(settings)),
            supports_custom_base_url=self.supports_custom_base_url,
        )

    def default_base_url(self, settings: Settings) -> str:
        raise NotImplementedError

    def default_api_key(self, settings: Settings) -> str:
        return ""

    def default_model(self, settings: Settings) -> str:
        return self.reference_models[0] if self.reference_models else ""

    def resolve_connection(self, settings: Settings, base_url: str | None, api_key: str | None) -> VlmConnection:
        return VlmConnection(
            provider=self.provider_id,
            base_url=(base_url or self.default_base_url(settings)).rstrip("/"),
            api_key=api_key if api_key is not None else self.default_api_key(settings),
        )

    def list_models(self, settings: Settings, base_url: str | None = None, api_key: str | None = None):
        raise NotImplementedError


def reference_model_infos(provider: str, models: tuple[str, ...], reason: str) -> list[VlmModelInfo]:
    return [
        VlmModelInfo(
            id=model,
            display_name=model,
            provider=provider,
            available=False,
            capabilities=["image_to_text"],
            source="reference_catalog",
            reason=reason,
        )
        for model in models
    ]


def merge_discovered_with_reference(
    provider: str,
    discovered_ids: list[str],
    reference_ids: tuple[str, ...],
) -> list[VlmModelInfo]:
    seen = set()
    models: list[VlmModelInfo] = []
    for model_id in discovered_ids:
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        models.append(
            VlmModelInfo(
                id=model_id,
                display_name=model_id,
                provider=provider,
                available=True,
                capabilities=["image_to_text"],
                source="provider_discovery",
            )
        )
    for model_id in reference_ids:
        if model_id in seen:
            continue
        models.append(
            VlmModelInfo(
                id=model_id,
                display_name=model_id,
                provider=provider,
                available=False,
                capabilities=["image_to_text"],
                source="reference_catalog",
                reason="Reference model; not returned by provider discovery.",
            )
        )
    return sorted(models, key=lambda model: (not model.available, model.id))
