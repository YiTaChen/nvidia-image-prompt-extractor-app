# NVIDIA Image Prompt Extractor App

Initial TDD implementation for two independent workflows:

- Upload an image and ask an NVIDIA vision-language model to generate an image prompt.
- Enter a prompt and ask Pollinations to generate an image.
- Run a capped prompt-refinement loop with prompt extraction, image generation, similarity scoring, prompt refinement, progress events, and cancellable background jobs.

## Current NVIDIA Endpoint Notes

`NVIDIA_BASE_URL` is used for the hosted VLM chat endpoint.

Hosted NVIDIA image generation is no longer part of the current development path. The app uses NVIDIA for VLM prompt extraction/refinement and Pollinations for prompt-to-image generation.

Example:

```env
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_VLM_MODEL=nvidia/llama-3.1-nemotron-nano-vl-8b-v1
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

## Test Fixture

The sample image provided by the user is stored at:

```text
backend/tests/fixtures/sample.png
```
