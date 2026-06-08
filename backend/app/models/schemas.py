from pydantic import BaseModel, Field


class PromptExtractionResult(BaseModel):
    prompt: str
    negative_prompt: str = ""
    analysis: dict[str, str] = Field(default_factory=dict)


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(min_length=1)
    negative_prompt: str = ""
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)
    seed: int | None = None


class ImageGenerationResult(BaseModel):
    image_base64: str
    mime_type: str = "image/png"
    model: str


class ErrorResponse(BaseModel):
    detail: str
