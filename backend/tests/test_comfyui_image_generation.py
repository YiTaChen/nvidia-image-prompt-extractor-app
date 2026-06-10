import base64

from fastapi.testclient import TestClient

from app.clients.comfyui_image_client import ComfyUIImageGenerationClient
from app.core.comfyui_workflows import get_workflow_definition, list_workflow_definitions, patch_workflow
from app.core.config import Settings, get_settings
from app.main import app
from app.models.schemas import ImageGenerationRequest


def _reset_settings():
    get_settings.cache_clear()
    settings = get_settings()
    settings.use_mock_nvidia = False
    settings.image_provider = "pollinations"
    settings.comfyui_base_url = "http://127.0.0.1:8188"
    settings.comfyui_api_key = ""
    settings.comfyui_workflow = "qwen_image_edit_plus_text_to_image"
    settings.comfyui_checkpoint = ""
    return settings


def test_comfyui_catalog_includes_text_to_image_template():
    workflows = list_workflow_definitions()

    text_workflow = next(workflow for workflow in workflows if workflow.id == "qwen_image_edit_plus_text_to_image")

    assert text_workflow.mode == "text_to_image"
    assert text_workflow.workflow_path.name == "qwen_image_edit_plus_text_to_image.workflow.json"
    assert text_workflow.bindings["prompt"] == "8.inputs.prompt"
    assert "image1" not in text_workflow.bindings


def test_comfyui_workflow_patcher_applies_prompt_size_seed_and_checkpoint():
    workflow = patch_workflow(
        "qwen_image_edit_plus_text_to_image",
        {
            "prompt": "a cinematic red door",
            "negative_prompt": "blur",
            "checkpoint": "custom.safetensors",
            "seed": 123,
            "width": 768,
            "height": 512,
            "filename_prefix": "codex-test",
        },
    )

    assert workflow["8"]["inputs"]["prompt"] == "a cinematic red door"
    assert workflow["2"]["inputs"]["prompt"] == "blur"
    assert workflow["9"]["inputs"]["ckpt_name"] == "custom.safetensors"
    assert workflow["5"]["inputs"]["seed"] == 123
    assert workflow["7"]["inputs"]["width"] == 768
    assert workflow["7"]["inputs"]["height"] == 512
    assert workflow["1"]["inputs"]["filename_prefix"] == "codex-test"


def test_comfyui_client_enqueues_polls_history_and_fetches_image(monkeypatch):
    settings = Settings(comfyui_base_url="http://comfy.test", comfyui_workflow="qwen_image_edit_plus_text_to_image")
    calls = []
    image_bytes = b"fake-png-bytes"

    class FakeResponse:
        def __init__(self, payload=None, content=b"", headers=None):
            self._payload = payload or {}
            self.content = content
            self.headers = headers or {"content-type": "image/png"}

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_post(url, **kwargs):
        calls.append(("post", url, kwargs))
        assert url == "http://comfy.test/prompt"
        payload = kwargs["json"]
        assert payload["prompt"]["8"]["inputs"]["prompt"] == "a small cabin"
        assert payload["prompt"]["2"]["inputs"]["prompt"] == "low quality"
        assert payload["prompt"]["7"]["inputs"]["width"] == 640
        assert payload["prompt"]["7"]["inputs"]["height"] == 512
        return FakeResponse({"prompt_id": "prompt-1"})

    def fake_get(url, **kwargs):
        calls.append(("get", url, kwargs))
        if url == "http://comfy.test/history/prompt-1":
            return FakeResponse(
                {
                    "prompt-1": {
                        "outputs": {
                            "1": {
                                "images": [
                                    {"filename": "ComfyUI_00001_.png", "subfolder": "", "type": "output"}
                                ]
                            }
                        }
                    }
                }
            )
        if url == "http://comfy.test/view":
            assert kwargs["params"] == {"filename": "ComfyUI_00001_.png", "subfolder": "", "type": "output"}
            return FakeResponse(content=image_bytes)
        raise AssertionError(url)

    monkeypatch.setattr("app.clients.comfyui_image_client.requests.post", fake_post)
    monkeypatch.setattr("app.clients.comfyui_image_client.requests.get", fake_get)

    result = ComfyUIImageGenerationClient(settings).generate_image(
        ImageGenerationRequest(
            prompt="a small cabin",
            negative_prompt="low quality",
            width=640,
            height=512,
            image_provider="comfyui",
        )
    )

    assert result.image_base64 == base64.b64encode(image_bytes).decode("ascii")
    assert result.model == "comfyui/qwen_image_edit_plus_text_to_image"
    assert result.provider == "comfyui"
    assert result.workflow == "qwen_image_edit_plus_text_to_image"
    assert [call[0] for call in calls] == ["post", "get", "get"]


def test_generate_image_route_can_select_comfyui(monkeypatch):
    _reset_settings()
    captured = {}

    def fake_generate_image(self, request):
        captured["request"] = request
        return {
            "image_base64": base64.b64encode(b"route-image").decode("ascii"),
            "mime_type": "image/png",
            "model": "comfyui/qwen_image_edit_plus_text_to_image",
            "provider": "comfyui",
            "workflow": "qwen_image_edit_plus_text_to_image",
        }

    monkeypatch.setattr(ComfyUIImageGenerationClient, "generate_image", fake_generate_image)
    client = TestClient(app)

    response = client.post(
        "/api/generate-image",
        json={
            "prompt": "a clean product render",
            "negative_prompt": "noise",
            "width": 768,
            "height": 768,
            "image_provider": "comfyui",
            "image_base_url": "http://comfy.local:8188",
            "image_workflow": "qwen_image_edit_plus_text_to_image",
            "image_model": "Qwen-Rapid-AIO-NSFW-v19.safetensors",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "comfyui"
    assert captured["request"].image_provider == "comfyui"
    assert captured["request"].image_base_url == "http://comfy.local:8188"
    assert captured["request"].image_model == "Qwen-Rapid-AIO-NSFW-v19.safetensors"


def test_image_generation_workflows_endpoint_lists_text_template():
    _reset_settings()
    client = TestClient(app)

    response = client.get("/api/image-generation/workflows")

    assert response.status_code == 200
    workflows = {workflow["id"]: workflow for workflow in response.json()["workflows"]}
    assert workflows["qwen_image_edit_plus_text_to_image"]["mode"] == "text_to_image"
    assert workflows["qwen_image_edit_plus_text_to_image"]["primary"] is True
