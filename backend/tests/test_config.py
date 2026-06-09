from app.core.config import Settings


def test_settings_defaults_are_safe(monkeypatch):
    monkeypatch.delenv("MAX_ITERATIONS", raising=False)
    settings = Settings(_env_file=None)

    assert settings.default_similarity_threshold == 80
    assert settings.max_iterations == 3
    assert settings.capped_max_iterations == 3
    assert settings.nvidia_base_url == "https://integrate.api.nvidia.com/v1"


def test_settings_detects_api_key():
    assert Settings(_env_file=None, nvidia_api_key="abc").has_nvidia_key is True
    assert Settings(_env_file=None, nvidia_api_key=" ").has_nvidia_key is False


def test_settings_can_use_separate_image_base_url():
    settings = Settings(
        _env_file=None,
        nvidia_base_url="https://integrate.api.nvidia.com/v1",
        nvidia_image_base_url="http://127.0.0.1:8001/v1",
    )

    assert settings.resolved_image_base_url == "http://127.0.0.1:8001/v1"
    assert settings.has_dedicated_image_base_url is True


def test_settings_caps_expensive_loop_iterations():
    settings = Settings(_env_file=None, max_iterations=12)

    assert settings.capped_max_iterations == 3
