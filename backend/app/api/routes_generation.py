from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.core.services import get_image_generation_client
from app.models.schemas import ImageGenerationRequest, ImageGenerationResult


router = APIRouter(prefix="/api", tags=["generation"])


@router.post("/generate-image", response_model=ImageGenerationResult)
async def generate_image(request: ImageGenerationRequest) -> ImageGenerationResult:
    settings = get_settings()
    client = get_image_generation_client(settings)
    try:
        return client.generate_image(request)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"NVIDIA image generation failed: {exc}") from exc
