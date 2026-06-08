# NVIDIA Image Prompt Extractor App - Agent Development Plan

## Project Summary

Build a web application that accepts an uploaded image, uses an NVIDIA vision-language model to extract a high-quality image-generation prompt, uses an NVIDIA image-generation model to generate a new image from that prompt, compares the generated image with the original image, and iteratively refines the prompt until the similarity score reaches the user-defined threshold.

Default similarity threshold: 80%.

The project must be developed with TDD. All core business logic should be testable without calling NVIDIA APIs by using mock clients.

## User Workflow

1. User uploads an original image.
2. User optionally adjusts similarity threshold, max iterations, generation size, seed, and negative prompt.
3. Backend validates and normalizes the image.
4. NVIDIA VLM generates the initial image prompt from the original image.
5. NVIDIA image-generation model generates an image from the prompt.
6. Backend calculates similarity between original image and generated image.
7. If score is greater than or equal to the threshold, stop and return the final result.
8. If score is lower than the threshold, send the original image, last generated image, last prompt, and score details to NVIDIA VLM to refine the prompt.
9. Repeat generation, scoring, and refinement until threshold is reached or max iterations is reached.
10. UI displays original image, best generated image, final prompt, final score, and iteration history.

## High-Level Architecture

```text
Frontend: React + Vite + TypeScript
Backend: FastAPI + Python
Persistence: SQLite + local filesystem
Realtime progress: Server-Sent Events or WebSocket
Testing: pytest, vitest, Playwright
Image processing: Pillow, OpenCV, scikit-image, imagehash
Optional semantic similarity: NVIDIA NVCLIP embeddings
```

## Recommended Directory Structure

```text
nvidia-image-prompt-extractor-app/
  .env
  .env.example
  README.md
  AGENT_PROJECT_PLAN.md
  backend/
    pyproject.toml
    app/
      main.py
      api/
        routes_upload.py
        routes_jobs.py
        routes_results.py
      core/
        config.py
        image_io.py
        prompt_extractor.py
        image_generator.py
        similarity.py
        refinement_loop.py
        job_store.py
      clients/
        nvidia_vlm_client.py
        nvidia_image_client.py
        nvidia_embedding_client.py
        mock_clients.py
      models/
        schemas.py
      storage/
        originals/
        generated/
        jobs.db
      tests/
        test_config.py
        test_image_io.py
        test_similarity.py
        test_refinement_loop.py
        test_api_jobs.py
  frontend/
    package.json
    vite.config.ts
    src/
      App.tsx
      api/client.ts
      components/
        ImageUploader.tsx
        ThresholdControl.tsx
        GenerationSettings.tsx
        RunTimeline.tsx
        ComparisonView.tsx
        PromptPanel.tsx
      tests/
        ImageUploader.test.tsx
        ThresholdControl.test.tsx
        RunTimeline.test.tsx
    e2e/
      prompt-loop.spec.ts
```

## Backend Responsibilities

### Configuration

Module: `backend/app/core/config.py`

Responsibilities:

- Read `.env`.
- Validate required environment variables.
- Provide default values for optional settings.
- Make model names configurable.
- Support mock mode for local tests.

Required settings:

```env
NVIDIA_API_KEY=
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_VLM_MODEL=nvidia/llama-3.1-nemotron-nano-vl-8b-v1
NVIDIA_IMAGE_MODEL=qwen/qwen-image
NVIDIA_EMBEDDING_MODEL=nvidia/nvclip
DEFAULT_SIMILARITY_THRESHOLD=80
MAX_ITERATIONS=5
IMAGE_OUTPUT_SIZE=1024
USE_MOCK_NVIDIA=false
```

### Image Input and Normalization

Module: `backend/app/core/image_io.py`

Responsibilities:

- Accept jpg, jpeg, png, and webp uploads.
- Reject unsupported formats.
- Enforce maximum file size.
- Normalize to RGB.
- Resize large images for API compatibility.
- Convert images to base64 data URLs.
- Save original and generated images to deterministic job paths.

### NVIDIA Vision Prompt Extraction

Module: `backend/app/clients/nvidia_vlm_client.py`

Responsibilities:

- Call NVIDIA OpenAI-compatible chat completions endpoint.
- Send one or more images as message content.
- Generate an initial prompt from the original image.
- Refine the prompt using original image, previous generated image, previous prompt, and similarity report.

Initial prompt output should be structured:

```json
{
  "prompt": "full image generation prompt",
  "negative_prompt": "things to avoid",
  "analysis": {
    "subject": "...",
    "composition": "...",
    "lighting": "...",
    "style": "...",
    "colors": "...",
    "camera": "...",
    "details": "..."
  }
}
```

### NVIDIA Image Generation

Module: `backend/app/clients/nvidia_image_client.py`

Responsibilities:

- Generate image from prompt.
- Accept generation settings such as width, height, seed, cfg scale, and negative prompt where supported by the selected model.
- Parse response image data.
- Save image to `backend/app/storage/generated/{job_id}/{iteration}.png`.

### Similarity Scoring

Module: `backend/app/core/similarity.py`

Responsibilities:

- Compare original and generated images.
- Return a score from 0 to 100.
- Return component scores for debugging and prompt refinement.

Recommended hybrid score:

```text
final_score =
  0.45 * embedding_similarity +
  0.35 * ssim_score +
  0.20 * perceptual_hash_score
```

Fallback when NVIDIA embedding model is unavailable:

```text
final_score =
  0.60 * ssim_score +
  0.40 * perceptual_hash_score
```

The scoring module must be deterministic and unit tested with local fixture images.

### Refinement Loop

Module: `backend/app/core/refinement_loop.py`

Responsibilities:

- Orchestrate the full iterative workflow.
- Stop when score reaches threshold.
- Stop when max iterations is reached.
- Preserve all attempts in job history.
- Track best attempt even if final threshold is not reached.
- Emit progress events for the UI.

Pseudo-code:

```python
def run_refinement_job(original_image, settings):
    prompt_result = vlm.generate_initial_prompt(original_image)
    best_attempt = None

    for iteration in range(1, settings.max_iterations + 1):
        generated_image = image_model.generate(prompt_result.prompt, settings)
        score = similarity.compare(original_image, generated_image)
        attempt = save_attempt(iteration, prompt_result, generated_image, score)
        best_attempt = max_by_score(best_attempt, attempt)

        if score.final_score >= settings.threshold:
            return completed(job_id, best_attempt, reached_threshold=True)

        prompt_result = vlm.refine_prompt(
            original_image=original_image,
            generated_image=generated_image,
            previous_prompt=prompt_result.prompt,
            similarity_report=score,
        )

    return completed(job_id, best_attempt, reached_threshold=False)
```

### Job Storage

Module: `backend/app/core/job_store.py`

Responsibilities:

- Create jobs.
- Update job status.
- Store iteration history.
- Store prompt, generated image path, score, and model metadata.
- Retrieve final result.

Recommended statuses:

```text
queued
running
completed
failed
cancelled
```

## API Endpoints

### Upload and Start Job

`POST /api/jobs`

Multipart form:

- `image`: uploaded image.
- `threshold`: optional number, default 80.
- `max_iterations`: optional integer, default 5.
- `seed`: optional integer.
- `negative_prompt`: optional string.

Response:

```json
{
  "job_id": "uuid",
  "status": "queued"
}
```

### Job Status

`GET /api/jobs/{job_id}`

Response:

```json
{
  "job_id": "uuid",
  "status": "running",
  "threshold": 80,
  "current_iteration": 2,
  "best_score": 76.4
}
```

### Job Events

`GET /api/jobs/{job_id}/events`

Use SSE or WebSocket to stream:

```json
{
  "type": "iteration_completed",
  "iteration": 2,
  "score": 76.4,
  "prompt": "..."
}
```

### Job Result

`GET /api/jobs/{job_id}/result`

Response:

```json
{
  "job_id": "uuid",
  "status": "completed",
  "reached_threshold": true,
  "threshold": 80,
  "best_score": 84.2,
  "final_prompt": "...",
  "original_image_url": "/api/files/originals/...",
  "best_image_url": "/api/files/generated/...",
  "iterations": [
    {
      "iteration": 1,
      "score": 71.2,
      "prompt": "...",
      "generated_image_url": "..."
    }
  ]
}
```

## Frontend Responsibilities

The first screen should be the actual app, not a landing page.

Required UI:

- Image uploader with preview.
- Similarity threshold control.
- Max iterations control.
- Optional advanced generation settings.
- Start button.
- Live progress timeline.
- Original vs generated image comparison.
- Final prompt panel with copy button.
- Iteration history with scores and thumbnails.
- Error states for invalid image, missing API key, NVIDIA API failure, and max iteration reached.

## TDD Development Plan

### Phase 1: Backend Foundation

Write tests first:

- `test_config.py`
  - loads required settings
  - applies default threshold
  - supports mock mode

- `test_image_io.py`
  - accepts valid image formats
  - rejects unsupported formats
  - converts to RGB
  - resizes oversized image
  - returns valid data URL

Implementation follows only after tests fail.

### Phase 2: Similarity Engine

Write tests first:

- identical images score near 100
- visually different images score lower
- final score remains in 0 to 100 range
- fallback scoring works without embedding client

Then implement SSIM, perceptual hash, and optional NVIDIA embedding similarity.

### Phase 3: Mock NVIDIA Clients

Write tests first:

- mock VLM returns structured initial prompt
- mock VLM returns refined prompt after low score
- mock image generator returns deterministic fixture image

Then implement `mock_clients.py`.

### Phase 4: Refinement Loop

Write tests first:

- stops immediately when first score passes threshold
- loops when score is below threshold
- stops at max iterations
- saves every iteration
- returns best attempt when threshold is not reached
- handles API failure and marks job failed

Then implement orchestration.

### Phase 5: API Layer

Write tests first:

- create job with image
- reject invalid upload
- get job status
- get job result
- stream or poll progress

Then implement FastAPI routes.

### Phase 6: Frontend

Write tests first:

- uploader shows preview
- threshold defaults to 80
- start button sends expected payload
- timeline displays iteration results
- comparison view displays original and generated image

Then implement components.

### Phase 7: End-to-End Tests

Use mock backend mode:

- upload fixture image
- run full prompt loop
- verify final result page
- verify iteration history

### Phase 8: NVIDIA Smoke Test

Only after mock tests pass:

- verify `.env` contains `NVIDIA_API_KEY`
- list or call configured VLM model
- generate one prompt from a small test image
- generate one image from a simple prompt
- calculate similarity

Do not run expensive or repeated NVIDIA calls during normal unit tests.

## Prompt Templates

### Initial Prompt Extraction

System message:

```text
You are an expert image prompt engineer. Analyze the uploaded image and produce a precise prompt for a text-to-image generation model. Preserve subject, composition, style, lighting, color palette, camera details, and important visual details. Return strict JSON only.
```

User message:

```text
Create a generation prompt for this image. The prompt should be detailed enough that an image generation model can recreate a visually similar image. Include a negative prompt.
```

### Prompt Refinement

System message:

```text
You are an expert image prompt engineer improving a text-to-image prompt through visual comparison. You will receive the original image, the last generated image, the previous prompt, and a similarity report. Rewrite the prompt to make the next generated image more visually similar to the original. Return strict JSON only.
```

User message:

```text
The last generated image did not meet the similarity threshold.

Previous prompt:
{previous_prompt}

Similarity report:
{similarity_report}

Compare the original image and the generated image. Identify missing or incorrect visual details, then return a better prompt and negative prompt.
```

## Acceptance Criteria

The project is complete when:

- User can upload an image.
- App extracts an initial prompt using NVIDIA VLM.
- App generates an image using NVIDIA image model.
- App computes similarity score.
- App refines prompt and loops when score is below threshold.
- App stops when threshold is reached or max iterations is reached.
- App displays final prompt, best image, final score, and iteration history.
- Backend unit tests pass.
- Frontend component tests pass.
- E2E test passes in mock mode.
- NVIDIA smoke test passes when a valid `.env` key is provided.

## Implementation Notes for Future Agents

- Never commit or print the real NVIDIA API key.
- Keep `.env` local and add `.env.example` for shared configuration.
- Use mock clients for TDD and CI.
- Keep NVIDIA model names configurable because API Catalog access can vary by account.
- Avoid hard-coding response shapes until the real endpoint has been smoke tested.
- Save all generated artifacts under backend storage paths, not in random temp directories.
- Prefer small fixture images for tests.
- Do not rely on visual similarity alone; use component scores to guide prompt refinement.
- Cap max iterations to control API cost.
- Log model name, iteration number, and score, but never log secrets.
