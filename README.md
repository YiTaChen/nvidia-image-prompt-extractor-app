# NVIDIA Image Prompt Extractor App

Initial TDD implementation for two independent workflows:

- Upload an image and ask an NVIDIA vision-language model to generate an image prompt.
- Enter a prompt and ask Pollinations or an NVIDIA image-generation endpoint to generate an image.
- Run a capped prompt-refinement loop that stops after at most 3 iterations by default.

The refinement loop and similarity scoring are planned in `AGENT_PROJECT_PLAN.md` but are not wired yet.

## Current NVIDIA Endpoint Notes

`NVIDIA_BASE_URL` is used for the hosted VLM chat endpoint.

`NVIDIA_IMAGE_BASE_URL` is required for real image generation. It must point to a Visual GenAI NIM `/v1` base URL, for example `http://localhost:8000/v1`.

The current hosted `https://integrate.api.nvidia.com/v1` endpoint successfully supports the configured VLM smoke test, but it does not expose the Visual GenAI `/images/generations` endpoint. NVIDIA's Qwen-Image docs show the image-generation client using a self-hosted NIM base URL such as `http://localhost:8000/v1` with model `qwen/qwen-image-2512`.

Several older hosted text-to-image endpoints, such as `stabilityai/sdxl-turbo`, `stabilityai/stable-diffusion-3-medium`, and `stabilityai/stable-diffusion-xl`, are supported by the backend as legacy candidates. They currently return 404 for this API key, which means the account does not have an active hosted/free text-to-image endpoint.

Example:

```env
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_IMAGE_BASE_URL=http://localhost:8000/v1
NVIDIA_VLM_MODEL=nvidia/llama-3.1-nemotron-nano-vl-8b-v1
NVIDIA_IMAGE_MODEL=qwen/qwen-image-2512
```

For development, the app defaults to Pollinations for prompt-to-image:

```env
IMAGE_PROVIDER=pollinations
POLLINATIONS_API_KEY=your_key_here
POLLINATIONS_MODEL=kontext
MAX_ITERATIONS=3
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

Hosted image endpoint smoke test:

```bash
cd ..
backend/.venv/bin/python backend/scripts/smoke_image_endpoints.py
```

Expected result when no hosted image endpoint is available:

```text
No hosted NVIDIA image-generation endpoint is available for this API key.
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

- Multipart form field: `image`
- Returns: `prompt`, `negative_prompt`, `analysis`

`POST /api/generate-image`

- JSON body: `prompt`, `negative_prompt`, `width`, `height`, `seed`
- Returns: `image_base64`, `mime_type`, `model`

`POST /api/refine-image`

- Multipart form fields: `image`, `threshold`, `max_iterations`
- Runs image prompt extraction, prompt-to-image generation, similarity scoring, and prompt refinement.
- Stops when the threshold is reached or the capped max iteration count is reached.

`GET /api/image-generation-config`

- Returns whether `NVIDIA_IMAGE_BASE_URL` has been configured for real image generation.

## Test Fixture

The sample image provided by the user is stored at:

```text
backend/tests/fixtures/sample.png
```
