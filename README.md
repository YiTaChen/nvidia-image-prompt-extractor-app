# Visual Prompt Studio

Full-stack AI web app for turning images into reusable generation prompts, generating images from prompts, and improving prompt quality through an automated refinement loop.

The project was built as a practical AI engineering portfolio piece: a FastAPI backend coordinates vision-language model prompt extraction, image generation providers, similarity scoring, and background jobs, while a React/Vite frontend gives users a clean workflow for comparing and refining results.

## Highlights

- Image-to-prompt workflow: upload an image and generate a structured prompt, negative prompt, and visual analysis with selectable VLM providers.
- Prompt-to-image workflow: generate images from text prompts through Pollinations or ComfyUI workflows.
- Iterative refinement loop: extract a prompt, generate an image, compare it with the original, refine the prompt, and repeat until the similarity threshold or capped iteration limit is reached.
- Multi-provider VLM support: NVIDIA hosted VLM, LM Studio, and Ollama, with provider-specific model discovery and safe API key handling.
- ComfyUI workflow integration: bundled workflow catalog and API-format workflow patching for prompt, negative prompt, seed, size, checkpoint, and output retrieval.
- Background job orchestration: queued refinement jobs, cancellable execution, deterministic local image storage, result lookup, and Server-Sent Events progress updates.
- Test-driven implementation: backend pytest coverage for API routes, clients, VLM parsing, image IO, similarity scoring, jobs, and refinement behavior; frontend Vitest coverage for core UI flows.

## Tech Stack

- Frontend: React 19, TypeScript, Vite, Vitest, Testing Library
- Backend: FastAPI, Pydantic Settings, Pillow, Requests, pytest
- AI providers: NVIDIA VLM endpoints, LM Studio, Ollama, Pollinations, ComfyUI
- Runtime patterns: REST APIs, multipart uploads, Server-Sent Events, background job state, local artifact storage

## How It Works

1. A user uploads a source image.
2. The backend normalizes the image and sends it to the selected VLM provider.
3. The VLM returns a detailed positive prompt, negative prompt, and visual analysis.
4. The user can generate a new image from that prompt with Pollinations or ComfyUI.
5. For refinement jobs, the backend scores generated images against the original and asks the VLM to improve the best prompt without dropping important subject, pose, clothing, or background details.

The refinement loop is capped at three iterations by default to keep image-generation cost and quota usage predictable.

## Main Features

### Image Prompt Extraction

`POST /api/extract-prompt`

Accepts an uploaded image and optional VLM provider settings. Returns:

- `prompt`
- `negative_prompt`
- `analysis`

### Image Generation

`POST /api/generate-image`

Accepts a prompt, negative prompt, size, seed, and optional provider overrides. Supported generation providers:

- `pollinations`
- `comfyui`

### Refinement Jobs

`POST /api/jobs`

Creates a background refinement job and stores the original image in:

```text
backend/app/storage/jobs/{job_id}/original.png
```

Additional job endpoints:

- `GET /api/jobs/{job_id}`
- `GET /api/jobs/{job_id}/events`
- `GET /api/jobs/{job_id}/result`
- `POST /api/jobs/{job_id}/cancel`

## Configuration

Create a `.env` file at the repository root. The app can run with local/mock settings for development, or with real provider credentials.

Example VLM configuration:

```env
VLM_PROVIDER=nvidia
VLM_MODEL=nvidia/nemotron-nano-12b-v2-vl
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_API_KEY=your_key_here
LM_STUDIO_BASE_URL=http://localhost:1234/v1
OLLAMA_BASE_URL=http://localhost:11434
```

Example image generation configuration:

```env
IMAGE_PROVIDER=pollinations
POLLINATIONS_API_KEY=your_key_here
POLLINATIONS_MODEL=kontext
MAX_ITERATIONS=3
```

Example ComfyUI configuration:

```env
IMAGE_PROVIDER=comfyui
COMFYUI_BASE_URL=http://127.0.0.1:8188
COMFYUI_API_KEY=
COMFYUI_WORKFLOW=qwen_image_edit_plus_text_to_image
COMFYUI_MULTI_IMAGE_EDIT_WORKFLOW=qwen_image_edit_plus_multi_image_edit
COMFYUI_CHECKPOINT=
COMFYUI_DENOISE_STRENGTH=0.55
COMFYUI_TIMEOUT_SECONDS=300
COMFYUI_POLL_INTERVAL_SECONDS=1
```

Bundled ComfyUI workflow metadata lives in:

```text
backend/app/workflows/comfyui/workflow_catalog.json
backend/app/workflows/comfyui/qwen_image_edit_plus_text_to_image.workflow.json
backend/app/workflows/comfyui/qwen_image_edit_plus_multi_image_edit.workflow.json
```

## Local Development

Backend:

```bash
cd backend
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest
```

Run the backend:

```bash
cd backend
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Run with mock AI clients:

```bash
cd backend
USE_MOCK_NVIDIA=true .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm test
npm run build
```

Run the frontend:

```bash
cd frontend
npm run dev -- --host 127.0.0.1
```

Open:

```text
http://127.0.0.1:5173/
```

## API Overview

- `GET /api/health`
- `POST /api/extract-prompt`
- `GET /api/vlm/providers`
- `POST /api/vlm/models`
- `POST /api/generate-image`
- `POST /api/refine-image`
- `POST /api/jobs`
- `GET /api/jobs/{job_id}`
- `GET /api/jobs/{job_id}/events`
- `GET /api/jobs/{job_id}/result`
- `POST /api/jobs/{job_id}/cancel`
- `GET /api/image-generation-config`
- `GET /api/image-generation/providers`
- `GET /api/image-generation/workflows`

## Resume Summary

Built a full-stack AI image workflow platform with FastAPI and React that extracts generation prompts from images using VLM providers, generates new images through Pollinations or ComfyUI, and runs an automated similarity-guided prompt refinement loop with background jobs, SSE progress updates, and comprehensive backend/frontend tests.
