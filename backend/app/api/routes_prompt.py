from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import get_settings
from app.core.image_io import ImageValidationError, bytes_to_data_url
from app.core.services import get_vision_client
from app.models.schemas import PromptExtractionResult


router = APIRouter(prefix="/api", tags=["prompt"])


@router.post("/extract-prompt", response_model=PromptExtractionResult)
async def extract_prompt(image: UploadFile = File(...)) -> PromptExtractionResult:
    content = await image.read()
    try:
        image_data_url = bytes_to_data_url(content)
    except ImageValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    settings = get_settings()
    client = get_vision_client(settings)
    try:
        return client.generate_initial_prompt(image_data_url)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"NVIDIA prompt extraction failed: {exc}") from exc
