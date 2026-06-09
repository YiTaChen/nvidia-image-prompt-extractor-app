from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.config import get_settings
from app.core.image_io import ImageValidationError
from app.core.refinement_loop import RefinementSettings, run_refinement_loop
from app.core.services import get_image_generation_client
from app.core.services import get_vision_client
from app.models.schemas import ImageGenerationRequest, ImageGenerationResult, RefinementResult


router = APIRouter(prefix="/api", tags=["generation"])


@router.post("/generate-image", response_model=ImageGenerationResult)
async def generate_image(request: ImageGenerationRequest) -> ImageGenerationResult:
    settings = get_settings()
    client = get_image_generation_client(settings)
    try:
        return client.generate_image(request)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Image generation failed: {exc}") from exc


@router.post("/refine-image", response_model=RefinementResult)
async def refine_image(
    image: UploadFile = File(...),
    threshold: float = Form(default=80),
    max_iterations: int | None = Form(default=None),
) -> RefinementResult:
    settings = get_settings()
    content = await image.read()
    try:
        loop_settings = RefinementSettings(
            threshold=max(0, min(100, threshold)),
            max_iterations=max(1, min(settings.capped_max_iterations, max_iterations or settings.capped_max_iterations)),
            width=settings.image_output_size,
            height=settings.image_output_size,
        )
        return run_refinement_loop(
            original_content=content,
            settings=loop_settings,
            vision_client=get_vision_client(settings),
            image_client=get_image_generation_client(settings),
        )
    except ImageValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Refinement loop failed: {exc}") from exc


@router.get("/image-generation-config")
async def image_generation_config() -> dict[str, str | bool]:
    settings = get_settings()
    return {
        "configured": settings.image_provider == "pollinations" and settings.has_pollinations_key
        or settings.has_dedicated_image_base_url,
        "provider": settings.image_provider,
        "model": settings.pollinations_model if settings.image_provider == "pollinations" else settings.nvidia_image_model,
        "base_url": "https://gen.pollinations.ai" if settings.image_provider == "pollinations" else settings.nvidia_image_base_url,
        "message": (
            "Image generation is configured."
            if (settings.image_provider == "pollinations" and settings.has_pollinations_key)
            or settings.has_dedicated_image_base_url
            else "Set POLLINATIONS_API_KEY or NVIDIA_IMAGE_BASE_URL before real image generation."
        ),
    }
