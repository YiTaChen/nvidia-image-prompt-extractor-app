from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.comfyui_workflows import list_workflow_definitions
from app.core.config import PROJECT_ROOT, get_settings
from app.core.image_io import ImageValidationError
from app.core.refinement_loop import RefinementSettings, run_refinement_loop
from app.core.services import get_image_generation_client
from app.core.services import get_vision_client
from app.models.schemas import (
    ImageGenerationRequest,
    ImageGenerationResult,
    ImageProviderInfo,
    ImageProvidersResponse,
    ImageWorkflowInfo,
    ImageWorkflowsResponse,
    RefinementResult,
)


router = APIRouter(prefix="/api", tags=["generation"])


@router.post("/generate-image", response_model=ImageGenerationResult)
async def generate_image(request: ImageGenerationRequest) -> ImageGenerationResult:
    settings = get_settings()
    try:
        client = get_image_generation_client(
            settings,
            provider=request.image_provider,
            base_url=request.image_base_url,
            api_key=request.image_api_key,
            workflow_id=request.image_workflow,
            model=request.image_model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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
    if settings.image_provider == "comfyui":
        return {
            "configured": bool(settings.comfyui_base_url.strip()),
            "provider": settings.image_provider,
            "model": settings.comfyui_checkpoint,
            "base_url": settings.comfyui_base_url,
            "message": "ComfyUI image generation is configured." if settings.comfyui_base_url else "Set COMFYUI_BASE_URL.",
        }
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


@router.get("/image-generation/providers", response_model=ImageProvidersResponse)
async def image_generation_providers() -> ImageProvidersResponse:
    settings = get_settings()
    return ImageProvidersResponse(
        providers=[
            ImageProviderInfo(
                id="pollinations",
                display_name="Pollinations",
                default_model=settings.pollinations_model,
                requires_api_key=True,
                api_key_configured=settings.has_pollinations_key,
                supports_custom_base_url=False,
            ),
            ImageProviderInfo(
                id="comfyui",
                display_name="ComfyUI",
                default_base_url=settings.comfyui_base_url,
                default_model=settings.comfyui_checkpoint,
                default_workflow=settings.comfyui_workflow,
                requires_api_key=False,
                api_key_configured=settings.has_comfyui_key,
                supports_custom_base_url=True,
                supports_workflows=True,
            ),
        ]
    )


@router.get("/image-generation/workflows", response_model=ImageWorkflowsResponse)
async def image_generation_workflows() -> ImageWorkflowsResponse:
    workflows = [
        ImageWorkflowInfo(
            id=workflow.id,
            display_name=workflow.display_name,
            mode=workflow.mode,
            description=workflow.description,
            workflow_path=str(workflow.workflow_path.relative_to(PROJECT_ROOT)),
            required_checkpoint=workflow.required_checkpoint,
            required_custom_nodes=workflow.required_custom_nodes or [],
            capabilities=workflow.capabilities,
            primary=workflow.primary,
        )
        for workflow in list_workflow_definitions()
    ]
    return ImageWorkflowsResponse(workflows=workflows)
