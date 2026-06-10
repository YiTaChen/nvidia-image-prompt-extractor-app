import base64
import time
import uuid
from typing import Any

import requests

from app.core.comfyui_workflows import patch_workflow
from app.core.config import Settings
from app.models.schemas import ImageGenerationRequest, ImageGenerationResult


class ComfyUIImageGenerationClient:
    def __init__(
        self,
        settings: Settings,
        base_url: str | None = None,
        api_key: str | None = None,
        workflow_id: str | None = None,
        checkpoint: str | None = None,
    ):
        self.settings = settings
        self.base_url = (base_url or settings.comfyui_base_url).rstrip("/")
        self.api_key = api_key if api_key is not None else settings.comfyui_api_key
        self.workflow_id = workflow_id or settings.comfyui_workflow
        self.checkpoint = checkpoint if checkpoint is not None else settings.comfyui_checkpoint

    def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        workflow_id = request.image_workflow or self.workflow_id
        checkpoint = request.image_model or self.checkpoint
        workflow = patch_workflow(
            workflow_id,
            {
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "checkpoint": checkpoint or None,
                "seed": request.seed,
                "width": request.width,
                "height": request.height,
                "filename_prefix": "Codex",
            },
        )
        prompt_id = self._enqueue_prompt(workflow)
        image_ref = self._wait_for_image(prompt_id)
        image_bytes, mime_type = self._fetch_image(image_ref)
        return ImageGenerationResult(
            image_base64=base64.b64encode(image_bytes).decode("ascii"),
            mime_type=mime_type,
            model=f"comfyui/{workflow_id}",
            provider="comfyui",
            workflow=workflow_id,
            seed=request.seed,
            mode="text_to_image",
            metadata={"prompt_id": prompt_id},
        )

    def _enqueue_prompt(self, workflow: dict[str, Any]) -> str:
        response = requests.post(
            f"{self.base_url}/prompt",
            headers=self._headers(),
            json={"prompt": workflow, "client_id": uuid.uuid4().hex},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        prompt_id = payload.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise RuntimeError("ComfyUI /prompt response did not include prompt_id.")
        return prompt_id

    def _wait_for_image(self, prompt_id: str) -> dict[str, str]:
        deadline = time.monotonic() + self.settings.comfyui_timeout_seconds
        while time.monotonic() <= deadline:
            response = requests.get(
                f"{self.base_url}/history/{prompt_id}",
                headers=self._headers(),
                timeout=30,
            )
            response.raise_for_status()
            image_ref = _first_history_image(response.json(), prompt_id)
            if image_ref:
                return image_ref
            time.sleep(self.settings.comfyui_poll_interval_seconds)
        raise RuntimeError(f"Timed out waiting for ComfyUI prompt {prompt_id}.")

    def _fetch_image(self, image_ref: dict[str, str]) -> tuple[bytes, str]:
        response = requests.get(
            f"{self.base_url}/view",
            headers=self._headers(),
            params={
                "filename": image_ref["filename"],
                "subfolder": image_ref.get("subfolder", ""),
                "type": image_ref.get("type", "output"),
            },
            timeout=120,
        )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "image/png").split(";")[0]
        if not content_type.startswith("image/"):
            raise RuntimeError(f"ComfyUI /view did not return an image. Content-Type: {content_type}")
        return response.content, content_type

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.api_key.strip():
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


def _first_history_image(payload: dict[str, Any], prompt_id: str) -> dict[str, str] | None:
    prompt_history = payload.get(prompt_id)
    if not isinstance(prompt_history, dict):
        return None
    outputs = prompt_history.get("outputs")
    if not isinstance(outputs, dict):
        return None
    for output in outputs.values():
        if not isinstance(output, dict):
            continue
        images = output.get("images")
        if not isinstance(images, list):
            continue
        for image in images:
            if isinstance(image, dict) and image.get("filename"):
                return {
                    "filename": str(image["filename"]),
                    "subfolder": str(image.get("subfolder", "")),
                    "type": str(image.get("type", "output")),
                }
    return None
