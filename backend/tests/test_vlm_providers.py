from pathlib import Path

from fastapi.testclient import TestClient

from app.clients.nvidia_vlm_client import NvidiaVisionClient
from app.clients.ollama_vlm_client import OllamaVisionClient
from app.clients.openai_compatible_vlm_client import OpenAiCompatibleVisionClient
from app.core.config import get_settings
from app.core.services import get_vision_client
from app.main import app
from app.models.schemas import PromptExtractionResult


FIXTURE = Path(__file__).parent / "fixtures" / "sample.png"


def _reset_settings():
    get_settings.cache_clear()
    settings = get_settings()
    settings.use_mock_nvidia = False
    settings.nvidia_api_key = ""
    settings.nvidia_base_url = "https://integrate.api.nvidia.com/v1"
    settings.nvidia_vlm_model = "nvidia/nemotron-nano-12b-v2-vl"
    settings.lm_studio_base_url = "http://localhost:1234/v1"
    settings.ollama_base_url = "http://localhost:11434"
    return settings


def test_vlm_providers_endpoint_lists_supported_providers():
    _reset_settings()
    client = TestClient(app)

    response = client.get("/api/vlm/providers")

    assert response.status_code == 200
    providers = {provider["id"]: provider for provider in response.json()["providers"]}
    assert {"nvidia", "lm_studio", "ollama"}.issubset(providers)
    assert providers["nvidia"]["api_key_configured"] is False
    assert providers["lm_studio"]["supports_custom_base_url"] is True


def test_nvidia_models_return_reference_catalog_without_key():
    _reset_settings()
    client = TestClient(app)

    response = client.post("/api/vlm/models", json={"provider": "nvidia"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["connection_status"] == "missing_key"
    assert payload["models"]
    assert payload["models"][0]["available"] is False
    assert any(model["id"] == "nvidia/nemotron-nano-12b-v2-vl" for model in payload["models"])


def test_nvidia_models_marks_discovered_models_available(monkeypatch):
    _reset_settings()

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": [
                    {"id": "01-ai/yi-large"},
                    {"id": "nvidia/nemotron-nano-12b-v2-vl"},
                    {"id": "meta/llama-3.2-90b-vision-instruct"},
                ]
            }

    monkeypatch.setattr("app.clients.vlm.nvidia_provider.requests.get", lambda *args, **kwargs: FakeResponse())
    client = TestClient(app)

    response = client.post("/api/vlm/models", json={"provider": "nvidia", "api_key": "runtime-key"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["connection_status"] == "ok"
    available = [model for model in payload["models"] if model["available"]]
    unavailable = [model for model in payload["models"] if not model["available"]]
    assert available
    assert unavailable
    assert all(model["id"] != "01-ai/yi-large" for model in payload["models"])
    assert payload["models"].index(available[-1]) < payload["models"].index(unavailable[0])


def test_lm_studio_models_return_reference_catalog_when_server_unavailable(monkeypatch):
    _reset_settings()

    def fail_get(*args, **kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr("app.clients.vlm.lm_studio_provider.requests.get", fail_get)
    client = TestClient(app)

    response = client.post("/api/vlm/models", json={"provider": "lm_studio"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["connection_status"] == "failed"
    assert payload["models"]
    assert all(model["source"] == "reference_catalog" for model in payload["models"])


def test_ollama_models_marks_local_tags_available(monkeypatch):
    _reset_settings()

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"models": [{"name": "llava:latest"}, {"name": "llama3.2-vision:11b"}]}

    monkeypatch.setattr("app.clients.vlm.ollama_provider.requests.get", lambda *args, **kwargs: FakeResponse())
    client = TestClient(app)

    response = client.post("/api/vlm/models", json={"provider": "ollama"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["connection_status"] == "ok"
    assert payload["models"][0]["available"] is True
    assert payload["models"][0]["id"] in {"llava:latest", "llama3.2-vision:11b"}


def test_extract_prompt_uses_runtime_nvidia_model_selection(monkeypatch):
    settings = _reset_settings()
    settings.nvidia_api_key = "env-key"
    captured = {}

    def fake_generate_initial_prompt(self, image_data_url):
        captured["api_key"] = self.api_key
        captured["base_url"] = self.base_url
        captured["model"] = self.model
        captured["image_data_url"] = image_data_url
        return PromptExtractionResult(prompt="selected prompt", negative_prompt="", analysis={})

    monkeypatch.setattr(NvidiaVisionClient, "generate_initial_prompt", fake_generate_initial_prompt)
    client = TestClient(app)

    with FIXTURE.open("rb") as image:
        response = client.post(
            "/api/extract-prompt",
            files={"image": ("sample.png", image, "image/png")},
            data={
                "vlm_provider": "nvidia",
                "vlm_model": "meta/llama-3.2-90b-vision-instruct",
                "vlm_base_url": "https://example.test/v1",
                "vlm_api_key": "runtime-key",
            },
        )

    assert response.status_code == 200
    assert response.json()["prompt"] == "selected prompt"
    assert captured["api_key"] == "runtime-key"
    assert captured["base_url"] == "https://example.test/v1"
    assert captured["model"] == "meta/llama-3.2-90b-vision-instruct"
    assert captured["image_data_url"].startswith("data:image/png;base64,")


def test_local_vlm_clients_can_use_generic_env_model_fallback():
    settings = _reset_settings()
    settings.vlm_provider = "lm_studio"
    settings.vlm_model = "local-vision-model"
    settings.lm_studio_vlm_model = ""

    lm_client = get_vision_client(settings)

    assert isinstance(lm_client, OpenAiCompatibleVisionClient)
    assert lm_client.model == "local-vision-model"

    settings.vlm_provider = "ollama"
    settings.vlm_model = "llava:latest"
    settings.ollama_vlm_model = ""

    ollama_client = get_vision_client(settings)

    assert isinstance(ollama_client, OllamaVisionClient)
    assert ollama_client.model == "llava:latest"
