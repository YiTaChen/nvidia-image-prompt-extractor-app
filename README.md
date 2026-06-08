# NVIDIA Image Prompt Extractor App

Initial TDD implementation for two independent workflows:

- Upload an image and ask an NVIDIA vision-language model to generate an image prompt.
- Enter a prompt and ask an NVIDIA image-generation endpoint to generate an image.

The refinement loop and similarity scoring are planned in `AGENT_PROJECT_PLAN.md` but are not wired yet.

## Current NVIDIA Endpoint Notes

`NVIDIA_BASE_URL` is used for the hosted VLM chat endpoint.

`NVIDIA_IMAGE_BASE_URL` is optional and should be used when image generation is served by a separate NVIDIA NIM endpoint. If it is empty, the backend tries `{NVIDIA_BASE_URL}/images/generations`.

The current hosted `https://integrate.api.nvidia.com/v1` endpoint successfully supports the configured VLM smoke test, but returned 404 for hosted image generation in this environment. `qwen/qwen-image` appears to require a downloadable/self-host NIM image endpoint.

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

- Multipart form field: `image`
- Returns: `prompt`, `negative_prompt`, `analysis`

`POST /api/generate-image`

- JSON body: `prompt`, `negative_prompt`, `width`, `height`, `seed`
- Returns: `image_base64`, `mime_type`, `model`

## Test Fixture

The sample image provided by the user is stored at:

```text
backend/tests/fixtures/sample.png
```
