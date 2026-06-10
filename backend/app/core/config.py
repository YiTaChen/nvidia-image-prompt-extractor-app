from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    vlm_provider: str = "nvidia"
    vlm_model: str = "nvidia/nemotron-nano-12b-v2-vl"
    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_image_base_url: str = ""
    nvidia_vlm_model: str = "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"
    nvidia_image_model: str = "qwen/qwen-image-2512"
    nvidia_embedding_model: str = "nvidia/nvclip"
    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com"
    gemini_vlm_model: str = "gemini-2.5-flash"
    lm_studio_base_url: str = "http://localhost:1234/v1"
    lm_studio_api_key: str = ""
    lm_studio_vlm_model: str = ""
    ollama_base_url: str = "http://localhost:11434"
    ollama_vlm_model: str = ""
    image_provider: str = "pollinations"
    pollinations_api_key: str = ""
    pollinations_model: str = "kontext"
    default_similarity_threshold: int = Field(default=80, ge=0, le=100)
    max_iterations: int = Field(default=3, ge=1, le=20)
    image_output_size: int = Field(default=1024, ge=256, le=2048)
    use_mock_nvidia: bool = False

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def has_nvidia_key(self) -> bool:
        return bool(self.nvidia_api_key.strip())

    @property
    def has_gemini_key(self) -> bool:
        return bool(self.gemini_api_key.strip())

    @property
    def resolved_image_base_url(self) -> str:
        return (self.nvidia_image_base_url or self.nvidia_base_url).rstrip("/")

    @property
    def has_dedicated_image_base_url(self) -> bool:
        return bool(self.nvidia_image_base_url.strip())

    @property
    def has_pollinations_key(self) -> bool:
        return bool(self.pollinations_api_key.strip())

    @property
    def capped_max_iterations(self) -> int:
        return max(1, min(3, self.max_iterations))


@lru_cache
def get_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    return Settings()
