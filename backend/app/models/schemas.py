from typing import Any

from pydantic import BaseModel, Field


class PromptExtractionResult(BaseModel):
    prompt: str
    negative_prompt: str = ""
    analysis: dict[str, Any] = Field(default_factory=dict)


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


class SimilarityScore(BaseModel):
    final_score: float
    histogram_score: float
    average_hash_score: float
    subject_histogram_score: float = 0
    subject_hash_score: float = 0
    subject_layout_score: float = 0
    edge_layout_score: float = 0
    critical_detail_score: float = 0


class RefinementAttempt(BaseModel):
    iteration: int
    prompt: str
    negative_prompt: str = ""
    generated_image_base64: str
    generated_image_mime_type: str
    generated_image_path: str | None = None
    score: SimilarityScore


class RefinementResult(BaseModel):
    reached_threshold: bool
    threshold: float
    max_iterations: int
    best_score: float
    final_prompt: str
    attempts: list[RefinementAttempt]


class ErrorResponse(BaseModel):
    detail: str


class VlmProviderInfo(BaseModel):
    id: str
    display_name: str
    default_base_url: str
    default_model: str
    requires_api_key: bool = False
    api_key_configured: bool = False
    supports_custom_base_url: bool = True


class VlmProvidersResponse(BaseModel):
    providers: list[VlmProviderInfo]


class VlmModelInfo(BaseModel):
    id: str
    display_name: str
    provider: str
    available: bool
    capabilities: list[str] = Field(default_factory=list)
    source: str
    reason: str | None = None


class VlmModelListRequest(BaseModel):
    provider: str
    base_url: str | None = None
    api_key: str | None = None


class VlmModelListResponse(BaseModel):
    provider: str
    connection_status: str
    message: str
    models: list[VlmModelInfo]


class JobCreateResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    threshold: float
    max_iterations: int
    current_iteration: int = 0
    best_score: float | None = None
    original_image_path: str | None = None
    error: str | None = None


class JobEvent(BaseModel):
    type: str
    job_id: str
    message: str
    iteration: int | None = None
    score: float | None = None
