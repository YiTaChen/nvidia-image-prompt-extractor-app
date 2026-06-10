# NVIDIA Image Prompt Extractor App

Initial TDD implementation for two independent workflows:

- Upload an image and ask a selectable vision-language model provider to generate an image prompt.
- Enter a prompt and ask Pollinations to generate an image. ComfyUI is planned as the next prompt-to-image provider.
- Run a capped prompt-refinement loop with prompt extraction, image generation, similarity scoring, prompt refinement, progress events, and cancellable background jobs.

## Current VLM Provider Notes

`NVIDIA_BASE_URL` is used for the hosted VLM chat endpoint.

Hosted NVIDIA image generation is no longer part of the current development path. The app uses NVIDIA for VLM prompt extraction/refinement and Pollinations for prompt-to-image generation.

## Multi-Provider VLM Selector

The image-to-prompt panel includes a two-level VLM selector:

- First dropdown: provider. Current implemented providers are NVIDIA, LM Studio, and Ollama.
- Second panel: provider-specific URL/key fields and a model dropdown.
- Model lists should show available discovered models first, followed by unavailable/reference models.
- Missing or invalid keys should not hide the reference model catalog.
- Secret fields must be password-masked and never returned raw from backend responses.

Gemini AI Studio remains planned in `AGENT_PROJECT_PLAN.md`.

Detailed implementation tasks are tracked in `AGENT_PROJECT_PLAN.md`.

Example:

```env
VLM_PROVIDER=nvidia
VLM_MODEL=nvidia/nemotron-nano-12b-v2-vl
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_VLM_MODEL=nvidia/llama-3.1-nemotron-nano-vl-8b-v1
LM_STUDIO_BASE_URL=http://localhost:1234/v1
LM_STUDIO_VLM_MODEL=
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_VLM_MODEL=
```

For development, the app defaults to Pollinations for prompt-to-image:

```env
IMAGE_PROVIDER=pollinations
POLLINATIONS_API_KEY=your_key_here
POLLINATIONS_MODEL=kontext
MAX_ITERATIONS=3
```

Planned ComfyUI prompt-to-image settings:

```env
IMAGE_PROVIDER=comfyui
COMFYUI_BASE_URL=http://127.0.0.1:8188
COMFYUI_API_KEY=
COMFYUI_WORKFLOW=qwen_image_edit_plus_text_to_image
COMFYUI_IMAGE_TO_IMAGE_WORKFLOW=image_to_image_basic
COMFYUI_MULTI_IMAGE_EDIT_WORKFLOW=qwen_image_edit_plus_multi_image_edit
COMFYUI_CHECKPOINT=
COMFYUI_DENOISE_STRENGTH=0.55
COMFYUI_TIMEOUT_SECONDS=300
COMFYUI_POLL_INTERVAL_SECONDS=1
```

The planned ComfyUI path will enqueue built-in text-to-image, image-to-image, and multi-image-edit workflows through ComfyUI, poll for completion, fetch the generated image through the backend, and keep generated files in the existing ignored local storage paths. Image-to-image support will upload an init image to ComfyUI, patch it into the workflow, and use denoise strength to control how much of the original composition is preserved. The bundled `qwen_image_edit_plus_multi_image_edit` workflow is marked for multi-reference Qwen image editing, not classic VAEEncode image-to-image. Detailed development phases are tracked in `AGENT_PROJECT_PLAN.md`.

Bundled ComfyUI workflow metadata lives in:

```text
backend/app/workflows/comfyui/workflow_catalog.json
backend/app/workflows/comfyui/qwen_image_edit_plus_text_to_image.workflow.json
backend/app/workflows/comfyui/qwen_image_edit_plus_multi_image_edit.workflow.json
```

The backend caps the refinement loop at 3 iterations even if a higher value is requested, to avoid burning free image-generation quota.

## Setup

Backend:

```bash
cd backend
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest
```

Frontend:

```bash
cd frontend
npm install
npm test
npm run build
```

## Run Locally

Backend with real NVIDIA settings:

```bash
cd backend
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Backend with mock NVIDIA clients:

```bash
cd backend
USE_MOCK_NVIDIA=true .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd frontend
npm run dev -- --host 127.0.0.1
```

Open:

```text
http://127.0.0.1:5173/
```

## Implemented Endpoints

`POST /api/extract-prompt`

- Multipart form fields: `image`, optional `vlm_provider`, `vlm_model`, `vlm_base_url`, `vlm_api_key`
- Returns: `prompt`, `negative_prompt`, `analysis`

`GET /api/vlm/providers`

- Returns selectable VLM providers and safe default metadata.
- Does not return raw API keys.

`POST /api/vlm/models`

- JSON body: `provider`, optional `base_url`, optional `api_key`
- Returns available provider-discovered models first, then built-in reference models.
- Keeps reference models visible when a key is missing or a local LM Studio/Ollama server is unavailable.

`POST /api/generate-image`

- JSON body: `prompt`, `negative_prompt`, `width`, `height`, `seed`
- Returns: `image_base64`, `mime_type`, `model`

`POST /api/refine-image`

- Multipart form fields: `image`, `threshold`, `max_iterations`
- Runs image prompt extraction, prompt-to-image generation, similarity scoring, and prompt refinement.
- Stops when the threshold is reached or the capped max iteration count is reached.

`POST /api/jobs`

- Multipart form fields: `image`, `threshold`, `max_iterations`
- Creates a queued background refinement job and immediately stores the normalized original image.
- Returns: `job_id`, `status`

`GET /api/jobs/{job_id}`

- Returns job status: `queued`, `running`, `completed`, `failed`, or `cancelled`.
- Includes current iteration, threshold, max iterations, progress events, and result metadata when available.

`GET /api/jobs/{job_id}/events`

- Server-Sent Events stream for realtime progress.
- Emits queued/running/iteration/completed/failed/cancelled events.

`GET /api/jobs/{job_id}/result`

- Returns the completed `RefinementResult`.
- Returns `409 Conflict` if the job is not completed yet.

`POST /api/jobs/{job_id}/cancel`

- Cancels a queued or running job.

`GET /api/image-generation-config`

- Returns the active prompt-to-image provider, model, base URL, and whether image generation is configured.

## Local Job Storage

Job files are stored in deterministic local paths and ignored by Git:

```text
backend/app/storage/jobs/{job_id}/original.png
backend/app/storage/jobs/{job_id}/generated/{iteration}.png
```

Each refinement attempt records the generated image storage path in the API result.

## Similarity Scoring

The score is foreground-person weighted. It still reports whole-image histogram and hash scores, but final similarity now also uses subject-region layout, edge layout, and `critical_detail_score` so matching backgrounds cannot hide incorrect people, hair color, clothing, or pose.

## Test Fixture

The sample image provided by the user is stored at:

```text
backend/tests/fixtures/sample.png
```
