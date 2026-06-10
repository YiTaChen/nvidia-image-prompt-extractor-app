from fastapi import APIRouter, HTTPException

from app.clients.vlm import get_vlm_provider, list_vlm_providers
from app.core.config import get_settings
from app.models.schemas import VlmModelListRequest, VlmModelListResponse, VlmProvidersResponse


router = APIRouter(prefix="/api/vlm", tags=["vlm"])


@router.get("/providers", response_model=VlmProvidersResponse)
async def providers() -> VlmProvidersResponse:
    settings = get_settings()
    return VlmProvidersResponse(
        providers=[provider.provider_info(settings) for provider in list_vlm_providers()],
    )


@router.post("/models", response_model=VlmModelListResponse)
async def models(request: VlmModelListRequest) -> VlmModelListResponse:
    settings = get_settings()
    try:
        provider = get_vlm_provider(request.provider)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown VLM provider: {request.provider}") from exc
    return provider.list_models(settings, base_url=request.base_url, api_key=request.api_key)
