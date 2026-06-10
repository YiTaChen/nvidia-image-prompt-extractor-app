from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.config import get_settings
from app.core.image_io import ImageValidationError, bytes_to_data_url
from app.core.services import get_vision_client
from app.models.schemas import PromptExtractionResult


router = APIRouter(prefix="/api", tags=["prompt"])


@router.post("/extract-prompt", response_model=PromptExtractionResult)
async def extract_prompt(
    image: UploadFile = File(...),
    vlm_provider: str | None = Form(None),
    vlm_model: str | None = Form(None),
    vlm_base_url: str | None = Form(None),
    vlm_api_key: str | None = Form(None),
) -> PromptExtractionResult:
    content = await image.read()
    try:
        image_data_url = bytes_to_data_url(content)
    except ImageValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    settings = get_settings()
    try:
        client = get_vision_client(
            settings,
            provider=vlm_provider,
            model=vlm_model,
            base_url=vlm_base_url,
            api_key=vlm_api_key,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        return client.generate_initial_prompt(image_data_url)
    except Exception as exc:
        provider_label = vlm_provider or settings.vlm_provider or "nvidia"
        raise HTTPException(status_code=502, detail=f"{provider_label} prompt extraction failed: {exc}") from exc
