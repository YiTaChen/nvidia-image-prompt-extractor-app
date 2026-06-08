from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_image_base_url: str = ""
    nvidia_vlm_model: str = "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"
    nvidia_image_model: str = "qwen/qwen-image"
    nvidia_embedding_model: str = "nvidia/nvclip"
    default_similarity_threshold: int = Field(default=80, ge=0, le=100)
    max_iterations: int = Field(default=5, ge=1, le=20)
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
    def resolved_image_base_url(self) -> str:
        return (self.nvidia_image_base_url or self.nvidia_base_url).rstrip("/")


@lru_cache
def get_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    return Settings()
