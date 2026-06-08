from app.core.config import Settings


def test_settings_defaults_are_safe():
    settings = Settings()

    assert settings.default_similarity_threshold == 80
    assert settings.max_iterations == 5
    assert settings.nvidia_base_url == "https://integrate.api.nvidia.com/v1"


def test_settings_detects_api_key():
    assert Settings(nvidia_api_key="abc").has_nvidia_key is True
    assert Settings(nvidia_api_key=" ").has_nvidia_key is False


def test_settings_can_use_separate_image_base_url():
    settings = Settings(
        nvidia_base_url="https://integrate.api.nvidia.com/v1",
        nvidia_image_base_url="http://127.0.0.1:8001/v1",
    )

    assert settings.resolved_image_base_url == "http://127.0.0.1:8001/v1"
