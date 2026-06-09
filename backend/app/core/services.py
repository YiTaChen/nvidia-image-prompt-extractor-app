from app.clients.mock_clients import MockImageGenerationClient, MockVisionClient
from app.clients.nvidia_image_client import NvidiaImageGenerationClient
from app.clients.nvidia_vlm_client import NvidiaVisionClient
from app.clients.pollinations_image_client import PollinationsImageGenerationClient
from app.core.config import Settings


def get_vision_client(settings: Settings):
    if settings.use_mock_nvidia:
        return MockVisionClient()
    return NvidiaVisionClient(settings)


def get_image_generation_client(settings: Settings):
    if settings.use_mock_nvidia:
        return MockImageGenerationClient()
    if settings.image_provider == "pollinations":
        return PollinationsImageGenerationClient(settings)
    return NvidiaImageGenerationClient(settings)
