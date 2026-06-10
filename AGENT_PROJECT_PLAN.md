# NVIDIA Image Prompt Extractor App - Agent Development Plan

## Project Summary

Build a web application that accepts an uploaded image, uses a selectable vision-language model to extract a high-quality image-generation prompt, uses a selectable prompt-to-image provider to generate a new image from that prompt, compares the generated image with the original image, and iteratively refines the prompt until the similarity score reaches the user-defined threshold.

Current provider decision: hosted NVIDIA image-generation endpoint development is cancelled. Do not add new hosted NVIDIA image endpoint work unless the user explicitly reopens it. NVIDIA remains available as a VLM provider for initial prompt extraction and prompt refinement. Pollinations remains the default prompt-to-image provider; ComfyUI is planned as the next prompt-to-image provider.

Default similarity threshold: 80%.

The project must be developed with TDD. All core business logic should be testable without calling NVIDIA APIs by using mock clients.

## User Workflow

1. User uploads an original image.
2. User optionally adjusts similarity threshold, max iterations, generation size, seed, and negative prompt.
3. Backend validates and normalizes the image.
4. Selected VLM generates the initial image prompt from the original image.
5. Selected prompt-to-image provider generates an image from the prompt.
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

### Multi-Provider VLM Selection

Goal: replace the hard-coded NVIDIA image-to-prompt model setting with a two-level provider/model selector so the same image prompt extraction workflow can use NVIDIA, Gemini AI Studio API, LM Studio, or Ollama.

This feature applies to image-to-text/image-to-prompt VLM usage first: initial prompt extraction, pose/action audit, and prompt refinement. Prompt-to-image generation remains a separate provider path.

Implementation status as of 2026-06-09:

- Completed: backend schemas, provider base class, provider registry, `GET /api/vlm/providers`, and `POST /api/vlm/models`.
- Completed: NVIDIA model discovery through `/models`, reference catalog fallback when the key is missing or discovery fails, and runtime NVIDIA model/base URL/key selection for `POST /api/extract-prompt`.
- Completed: LM Studio reference catalog and OpenAI-compatible `/models` discovery, plus an OpenAI-compatible VLM prompt client for clone users with local LM Studio servers.
- Completed: Ollama reference catalog and `/api/tags` discovery, plus an Ollama `/api/chat` VLM prompt client for clone users with local Ollama servers.
- Completed: image-to-prompt frontend selector with provider dropdown, URL field, password-masked key field, model dropdown, model refresh, and selected provider/model submission.
- Completed: unit/API tests for provider listing, model discovery fallback, available-model sorting, Ollama tag mapping, LM Studio failure fallback, and runtime NVIDIA extraction selection.
- Not completed: Gemini provider adapter.
- Not completed: `/api/vlm/test-connection`.
- Not completed: passing VLM selection into background jobs/refinement loop UI and storing provider/model metadata in job attempts.
- Not completed: dedicated frontend interaction tests for the selector.
- Not completed: real LM Studio/Ollama smoke tests, because no local test servers are currently available in this environment.

#### UX Requirements

First-level dropdown: platform/provider.

Initial provider options:

- `nvidia`: NVIDIA hosted OpenAI-compatible API.
- `gemini`: Google Gemini AI Studio API.
- `lm_studio`: local LM Studio OpenAI-compatible server.
- `ollama`: local Ollama server.

Second-level panel changes based on selected provider:

- Connection URL input where the provider supports custom URLs.
- API key/password input where required or optional.
- Model dropdown with available models first, then unavailable/reference models.
- Connection test/status badge.
- Read-only source labels showing whether a value came from `.env`, user runtime override, provider discovery, or built-in reference catalog.
- Secret textboxes must use password masking by default and expose a reveal toggle only if needed.
- The user can choose `.env` values or type temporary runtime values. Runtime values must not be written to `.env` unless a future explicit save feature is requested.

Model dropdown behavior:

- Always render a model list even when the key or local server is unavailable.
- Available models are sorted above unavailable/reference models.
- Unavailable/reference models remain selectable only if the user explicitly enables advanced/manual mode, or they appear disabled with a reason.
- Each model row should show provider, model id, availability, capability hints, and an optional failure reason.

Minimum model metadata shape:

```json
{
  "id": "nvidia/nemotron-nano-12b-v2-vl",
  "display_name": "Nemotron Nano 12B VL",
  "provider": "nvidia",
  "available": true,
  "capabilities": ["image_to_text", "chat_completions"],
  "source": "provider_discovery",
  "reason": null
}
```

#### Backend Design

Add a provider abstraction for image-to-prompt VLMs.

Recommended modules:

```text
backend/app/clients/vlm/
  __init__.py
  base.py
  registry.py
  nvidia_provider.py
  gemini_provider.py
  lm_studio_provider.py
  ollama_provider.py
```

Provider interface:

```python
class VlmProvider:
    provider_id: str

    def list_models(self, connection: VlmConnectionSettings) -> list[VlmModelInfo]:
        ...

    def generate_initial_prompt(self, image_data_url: str, settings: VlmRunSettings) -> PromptExtractionResult:
        ...

    def classify_pose_and_motion(self, image_data_url: str, settings: VlmRunSettings) -> dict:
        ...

    def refine_prompt(
        self,
        original_image_data_url: str,
        generated_image: Image.Image,
        previous_prompt: str,
        previous_negative_prompt: str,
        similarity_report: SimilarityScore,
        settings: VlmRunSettings,
    ) -> PromptExtractionResult:
        ...
```

Shared schemas to add under `backend/app/models/schemas.py`:

- `VlmProviderId`: enum/string literal for `nvidia`, `gemini`, `lm_studio`, `ollama`.
- `VlmConnectionSettings`: provider id, base URL, API key redacted flag, selected model, timeout.
- `VlmModelInfo`: id, display name, provider, available, capabilities, source, reason.
- `VlmProviderConfigResponse`: provider metadata, env defaults, required fields, optional fields.
- `VlmModelListResponse`: selected provider, connection status, available models, unavailable models.
- `VlmSelection`: provider, model, base URL override, API key override.

Connection settings:

- Never return raw API keys from backend responses.
- Allow runtime API key overrides only for the current request/session.
- Prefer `.env` values when runtime override is absent.
- Log provider id, model id, URL host, and availability only; never log secrets.

Provider default settings:

```env
VLM_PROVIDER=nvidia
VLM_MODEL=nvidia/nemotron-nano-12b-v2-vl

NVIDIA_API_KEY=
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_VLM_MODEL=nvidia/nemotron-nano-12b-v2-vl

GEMINI_API_KEY=
GEMINI_BASE_URL=https://generativelanguage.googleapis.com
GEMINI_VLM_MODEL=gemini-2.5-flash

LM_STUDIO_BASE_URL=http://localhost:1234/v1
LM_STUDIO_API_KEY=
LM_STUDIO_VLM_MODEL=

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_VLM_MODEL=
```

Provider model discovery:

- NVIDIA: call `/models` when key is available, merge results with built-in reference VLM candidates.
- Gemini: call the model listing endpoint when key is available, merge with built-in Gemini reference VLM candidates.
- LM Studio: call OpenAI-compatible `/models` on the configured local URL, merge with manual/reference entries.
- Ollama: call local tags/model listing endpoint, merge with known vision-capable local model references.

When provider discovery fails:

- Return `connection_status: failed`.
- Include failure reason in a safe, non-secret message.
- Still return built-in reference models with `available: false` and `source: reference_catalog`.

Suggested API endpoints:

`GET /api/vlm/providers`

- Returns provider list and field metadata for the two-level selector.
- Does not test remote connectivity.

`POST /api/vlm/models`

- Body: provider, optional base URL override, optional API key override.
- Returns available and unavailable/reference models.

`POST /api/vlm/test-connection`

- Body: provider, base URL override, API key override, model.
- Performs a lightweight model listing or metadata check.
- Returns safe status and reason.

`POST /api/extract-prompt`

- Extend multipart form with optional `vlm_provider`, `vlm_model`, `vlm_base_url`, and `vlm_api_key`.
- Uses `.env` fallback when fields are absent.

`POST /api/jobs`

- Extend multipart form with the same VLM selection fields.
- Store provider/model metadata in job attempts.

#### Frontend Design

Recommended components:

```text
frontend/src/components/VlmProviderSelector.tsx
frontend/src/components/VlmConnectionPanel.tsx
frontend/src/components/VlmModelSelect.tsx
frontend/src/components/SecretInput.tsx
frontend/src/api/vlm.ts
```

UI flow:

1. User opens image-to-prompt or refinement loop panel.
2. First dropdown selects provider.
3. Provider-specific connection fields render below:
   - NVIDIA: base URL, API key, model.
   - Gemini AI Studio: base URL, API key, model.
   - LM Studio: local URL, optional API key, model.
   - Ollama: local URL, model.
4. User can click refresh models.
5. Model dropdown shows available models first, unavailable/reference models below.
6. User can run image-to-prompt with selected provider/model.

Frontend state rules:

- Do not store typed API keys in localStorage by default.
- Keep temporary key values in React state only.
- Use `type="password"` for key fields.
- Show key source as `.env configured`, `runtime override`, or `missing`.
- Display provider/model used in every prompt extraction result.

#### TDD Plan

Backend tests first:

- `test_vlm_provider_registry.py`
  - lists all configured provider ids.
  - resolves default provider from `.env`.
  - rejects unknown provider ids.

- `test_vlm_model_catalog.py`
  - returns reference models when API key is missing.
  - marks discovered models as available.
  - sorts available models before unavailable models.
  - preserves unavailable models with reasons.

- `test_vlm_connection_settings.py`
  - uses `.env` when runtime override is absent.
  - uses runtime override without mutating `.env`.
  - redacts keys from responses/loggable payloads.

- `test_vlm_routes.py`
  - `GET /api/vlm/providers` returns provider metadata.
  - `POST /api/vlm/models` works with mock provider discovery.
  - `POST /api/vlm/test-connection` returns safe success/failure.
  - `POST /api/extract-prompt` can select a mock provider/model.

- Provider-specific client tests:
  - NVIDIA maps `/models` response to `VlmModelInfo`.
  - Gemini maps model listing response to `VlmModelInfo`.
  - LM Studio maps OpenAI-compatible `/models` response.
  - Ollama maps local model tags/list response.
  - each provider can convert local image data into its required request format.

Frontend tests first:

- provider dropdown defaults to `.env` provider.
- changing provider updates connection fields.
- key input is password-masked.
- model dropdown sorts available before unavailable.
- refresh models calls `/api/vlm/models`.
- selected provider/model are included in extract-prompt and job requests.
- unavailable models render with disabled styling or manual-mode gating.

#### Development Phases

Phase MP-1: Backend contracts and mock registry.

- Add schemas, provider interface, registry, and mock provider.
- Add provider/model API routes.
- No real external calls yet.

Phase MP-2: NVIDIA provider adapter.

- Move existing NVIDIA VLM behavior behind the provider interface.
- Implement NVIDIA model discovery via `/models`.
- Keep pose audit and prompt composition behavior intact.
- Add tests for currently observed candidate models such as `nvidia/llama-3.1-nemotron-nano-vl-8b-v1`, `nvidia/nemotron-nano-12b-v2-vl`, and `meta/llama-3.2-90b-vision-instruct` as reference entries, but do not assume every account can invoke all of them.

Phase MP-3: Gemini, LM Studio, and Ollama provider adapters.

- Add reference catalogs first.
- Add discovery calls and safe error handling.
- Add smoke scripts for each provider, guarded by explicit environment availability.

Phase MP-4: Frontend selector UI.

- Build provider dropdown, connection panels, model dropdown, key fields, and refresh/test buttons.
- Wire selected provider/model to prompt extraction first.

Phase MP-5: Job integration.

- Pass VLM selection into `/api/jobs`.
- Store provider/model metadata in job status/result/attempts.
- Display provider/model in timeline and result panels.

Phase MP-6: Real smoke verification.

- Verify NVIDIA with `.env`.
- Verify Gemini only if `GEMINI_API_KEY` is present.
- Verify LM Studio only if local server responds.
- Verify Ollama only if local server responds.
- Never fail unit tests because a real provider is unavailable.

#### Acceptance Criteria

- The app has a first-level provider dropdown.
- The second-level panel changes required connection fields per provider.
- API keys are masked and never returned raw from backend responses.
- The model dropdown shows available models first and unavailable/reference models second.
- Missing/invalid key still shows reference model lists.
- User can run image-to-prompt with selected provider/model.
- Jobs record the VLM provider/model used.
- All provider discovery and routing behavior is covered by mock tests.
- Real provider smoke tests are opt-in and never required for normal CI.

### Prompt-To-Image Generation

Primary module: `backend/app/clients/pollinations_image_client.py`

Legacy/self-hosted NVIDIA module: `backend/app/clients/nvidia_image_client.py`

Responsibilities:

- Generate image from prompt.
- Accept generation settings such as width, height, seed, cfg scale, and negative prompt where supported by the selected model.
- Parse response image data.
- Save image to `backend/app/storage/generated/{job_id}/{iteration}.png`.
- Keep Pollinations as the current development path.
- Do not implement or prioritize hosted NVIDIA image endpoints.

### Prompt-To-Image Provider Selection and ComfyUI

Goal: add ComfyUI as a second prompt-to-image provider beside Pollinations, with provider selection available in the standalone prompt-to-image panel and later reused by the refinement loop/background jobs.

Implementation status as of 2026-06-10:

- Completed: Pollinations prompt-to-image client and `/api/generate-image`.
- Completed: image generation config endpoint for the current active provider.
- Planned: provider abstraction/registry for prompt-to-image generation.
- Planned: ComfyUI client, workflow templating, model/workflow discovery, frontend provider selector, and job integration.
- Planned: real ComfyUI smoke test only when a local ComfyUI server is available.

#### ComfyUI API Assumptions

The backend should call ComfyUI through the user's local or remote ComfyUI HTTP API. Initial implementation should support:

- `GET /system_stats` for connection health and server metadata.
- `GET /models` and `GET /models/{folder}` for model folder/checkpoint discovery when available.
- `POST /prompt` to enqueue a workflow graph with a generated `client_id`.
- `GET /history/{prompt_id}` to poll for completed outputs.
- `GET /view?filename=...&subfolder=...&type=output` to fetch generated image bytes.
- Optional `GET /ws?clientId=...` for live execution progress after the synchronous polling path works.
- Optional `POST /interrupt` for cancellation after job queue integration.

#### Configuration

Add settings to `backend/app/core/config.py`:

```env
IMAGE_PROVIDER=pollinations

POLLINATIONS_API_KEY=
POLLINATIONS_MODEL=kontext

COMFYUI_BASE_URL=http://127.0.0.1:8188
COMFYUI_API_KEY=
COMFYUI_WORKFLOW=text_to_image_basic
COMFYUI_CHECKPOINT=
COMFYUI_POSITIVE_NODE_ID=
COMFYUI_NEGATIVE_NODE_ID=
COMFYUI_SEED_NODE_ID=
COMFYUI_WIDTH_NODE_ID=
COMFYUI_HEIGHT_NODE_ID=
COMFYUI_SAVE_NODE_ID=
COMFYUI_TIMEOUT_SECONDS=300
COMFYUI_POLL_INTERVAL_SECONDS=1
```

Settings rules:

- `IMAGE_PROVIDER=pollinations` remains the default.
- `IMAGE_PROVIDER=comfyui` selects ComfyUI for server-side generation.
- ComfyUI API keys are optional because local ComfyUI usually has no auth, but the field allows reverse-proxy/API-gateway deployments.
- Runtime overrides from the UI must be request-scoped and must not write to `.env`.
- Never return raw `COMFYUI_API_KEY` in any API response.
- Default ComfyUI URL should be loopback-only. Document that public ComfyUI instances should be protected by auth or a private network.

#### Backend Design

Add a prompt-to-image provider abstraction parallel to the VLM selector, but keep it independent from VLM providers.

Recommended modules:

```text
backend/app/clients/image_generation/
  __init__.py
  base.py
  registry.py
  pollinations_provider.py
  comfyui_provider.py

backend/app/clients/comfyui_image_client.py
backend/app/core/comfyui_workflows.py
backend/app/workflows/comfyui/text_to_image_basic.workflow.json
```

Provider interface:

```python
class ImageGenerationProvider:
    provider_id: str

    def provider_info(self, settings: Settings) -> ImageProviderInfo:
        ...

    def list_models(self, settings: Settings, connection: ImageProviderConnection) -> ImageModelListResponse:
        ...

    def list_workflows(self, settings: Settings) -> ImageWorkflowListResponse:
        ...

    def generate_image(self, request: ImageGenerationRequest, selection: ImageProviderSelection) -> ImageGenerationResult:
        ...
```

ComfyUI client responsibilities:

- Build request-scoped `client_id`.
- Load a workflow template JSON by id.
- Patch prompt, negative prompt, seed, width, height, checkpoint/model, and optional sampler settings into known node ids.
- Submit `POST /prompt` with `{ "prompt": workflow, "client_id": client_id }`.
- Parse `prompt_id`.
- Poll `/history/{prompt_id}` until a `SaveImage` output is available or timeout is reached.
- Fetch the first generated image through `/view`.
- Return `ImageGenerationResult(image_base64, mime_type, model="comfyui/{workflow_or_checkpoint}")`.
- Preserve raw workflow output metadata internally for debugging, but do not expose local filesystem paths from ComfyUI.

Workflow templating rules:

- Store safe built-in workflow templates in git under `backend/app/workflows/comfyui/`.
- Start with one basic text-to-image workflow using a checkpoint loader, positive/negative CLIP text encode nodes, sampler, VAE decode, and SaveImage.
- Treat custom uploaded workflow JSON as a future feature; do not allow arbitrary workflow uploads in the first ComfyUI milestone.
- Node ids must be configurable per workflow manifest, not hard-coded in the client.
- Validate workflow JSON before submit: must be a dict, must contain configured node ids, and must not include unexpected file upload/path mutation behavior.
- Keep ComfyUI output images out of git by saving app copies under existing ignored storage paths.

Suggested workflow manifest shape:

```json
{
  "id": "text_to_image_basic",
  "display_name": "Basic Text to Image",
  "workflow_path": "backend/app/workflows/comfyui/text_to_image_basic.workflow.json",
  "bindings": {
    "prompt": "6.inputs.text",
    "negative_prompt": "7.inputs.text",
    "seed": "3.inputs.seed",
    "width": "5.inputs.width",
    "height": "5.inputs.height",
    "checkpoint": "4.inputs.ckpt_name",
    "save": "9"
  },
  "capabilities": ["text_to_image", "negative_prompt", "seed", "size"]
}
```

#### Schemas

Add schemas under `backend/app/models/schemas.py`:

- `ImageProviderInfo`: provider id, display name, default URL, requires API key, API key configured flag, supports model discovery, supports workflows.
- `ImageProviderSelection`: provider id, base URL override, API key override, model/checkpoint, workflow id, workflow parameter overrides.
- `ImageModelInfo`: id, display name, provider, available, source, reason.
- `ImageModelListRequest` and `ImageModelListResponse`.
- `ImageWorkflowInfo` and `ImageWorkflowListResponse`.
- Extend `ImageGenerationRequest` with optional provider selection fields, or add a wrapper request if preserving backward compatibility is cleaner.
- Extend `ImageGenerationResult` with optional `provider`, `workflow`, `seed`, and `metadata` fields when useful.

#### API Endpoints

Keep existing `POST /api/generate-image` backward compatible.

Add:

`GET /api/image-generation/providers`

- Returns Pollinations and ComfyUI provider metadata.
- Does not test remote connectivity.
- Does not return secrets.

`POST /api/image-generation/models`

- Body: provider, optional base URL override, optional API key override.
- Pollinations: return configured/reference models.
- ComfyUI: call `/models/checkpoints` when possible; if not available, show reference/manual placeholder entries.

`GET /api/image-generation/workflows`

- Returns built-in ComfyUI workflow templates and capabilities.

`POST /api/image-generation/test-connection`

- Pollinations: checks whether API key appears configured without spending generation credits.
- ComfyUI: calls `/system_stats` and optionally `/models/checkpoints`.
- Returns safe status and message.

Extend `POST /api/generate-image`

- Accept optional `image_provider`, `image_base_url`, `image_api_key`, `image_model`, `image_workflow`, and generation controls.
- Use `.env` fallback when runtime values are absent.
- For ComfyUI, return a normal `ImageGenerationResult` so existing UI and refinement code can render it.

Extend `POST /api/jobs` and `POST /api/refine-image` in a later phase

- Accept the same image-generation selection fields.
- Store image provider/model/workflow metadata per attempt.
- Make job cancellation call the ComfyUI interrupt path when the current attempt is running in ComfyUI.

#### Frontend Design

Add a prompt-to-image provider selector to `PromptImagePanel`.

Recommended components:

```text
frontend/src/components/ImageGenerationProviderSelector.tsx
frontend/src/components/ImageGenerationModelSelect.tsx
frontend/src/components/ComfyWorkflowSelect.tsx
frontend/src/api/imageGeneration.ts
```

UI requirements:

- First dropdown: `Pollinations`, `ComfyUI`.
- Pollinations panel: API key password field, model dropdown/text field.
- ComfyUI panel: base URL field, optional API key password field, workflow dropdown, checkpoint/model dropdown, refresh button, connection status.
- Keep prompt and negative prompt textareas unchanged.
- Pass selected provider settings into `generateImage`.
- Do not store typed API keys in localStorage.
- Show provider/model/workflow used under the generated preview.
- If ComfyUI is unavailable, keep the panel usable with a clear status and manual checkpoint/workflow selection.

#### TDD Plan

Backend tests first:

- `test_image_generation_provider_registry.py`
  - lists `pollinations` and `comfyui`.
  - resolves `.env` defaults.
  - rejects unknown providers.

- `test_comfyui_workflows.py`
  - loads built-in workflow manifest.
  - patches prompt, negative prompt, seed, width, height, and checkpoint into the expected node paths.
  - rejects missing node bindings.
  - rejects malformed workflow JSON.

- `test_comfyui_client.py`
  - posts workflow to `/prompt` with `client_id`.
  - parses returned `prompt_id`.
  - polls `/history/{prompt_id}` until output appears.
  - fetches `/view` image bytes and returns base64 image.
  - handles timeout with a clear error.
  - handles queue/server errors without leaking secrets.

- `test_image_generation_routes.py`
  - `GET /api/image-generation/providers` returns safe metadata.
  - `POST /api/image-generation/models` maps ComfyUI checkpoint list.
  - `GET /api/image-generation/workflows` returns built-in workflows.
  - `POST /api/generate-image` can select ComfyUI using mocked HTTP calls.
  - backward-compatible Pollinations generation still works.

- `test_refinement_loop_image_provider_selection.py`
  - later phase: refinement loop uses selected image provider.
  - job attempt metadata records provider/model/workflow.
  - cancellation calls ComfyUI interrupt when applicable.

Frontend tests first:

- provider dropdown defaults to `.env` active provider.
- changing provider swaps connection fields.
- ComfyUI key field is password-masked.
- refresh models calls `/api/image-generation/models`.
- workflow dropdown renders built-in workflows.
- `generateImage` includes selected provider fields.
- generated preview displays provider/model/workflow metadata.

#### Development Phases

Phase IMG-MP-1: Backend provider contracts.

- Add schemas, provider interface, registry, and provider metadata endpoints.
- Keep `/api/generate-image` backward compatible.
- Add tests before implementation.

Phase IMG-MP-2: Pollinations adapter migration.

- Wrap existing `PollinationsImageGenerationClient` behind the image provider interface.
- Preserve current behavior and tests.
- Ensure no additional Pollinations calls are made during provider discovery.

Phase IMG-MP-3: ComfyUI workflow foundation.

- Add built-in workflow JSON and manifest.
- Add workflow loader/patcher with node-binding validation.
- Add tests for prompt/negative/seed/size/checkpoint patching.

Phase IMG-MP-4: ComfyUI HTTP client.

- Implement health/model discovery using ComfyUI HTTP endpoints.
- Implement synchronous enqueue/poll/fetch generation.
- Add timeout and safe error handling.
- Add opt-in smoke script guarded by `COMFYUI_BASE_URL`.

Phase IMG-MP-5: Frontend provider selector.

- Add image-generation provider selector to prompt-to-image panel.
- Wire selected provider/model/workflow to `generateImage`.
- Display provider metadata in results.

Phase IMG-MP-6: Refinement loop and job integration.

- Pass selected image-generation provider into `/api/refine-image` and `/api/jobs`.
- Persist provider/model/workflow metadata in each attempt.
- Add cancellation bridge to ComfyUI interrupt for active ComfyUI jobs.

Phase IMG-MP-7: Real local validation.

- Verify Pollinations remains functional.
- Verify ComfyUI only when local server responds.
- Verify generated image retrieval and storage path.
- Never require real ComfyUI in normal unit tests or CI.

#### Acceptance Criteria

- The app can choose Pollinations or ComfyUI for prompt-to-image generation.
- Existing Pollinations behavior remains backward compatible.
- ComfyUI can generate an image from prompt/negative prompt/seed/size using a built-in workflow.
- ComfyUI generated images are fetched through the backend and returned in the current `ImageGenerationResult` shape.
- Missing ComfyUI server does not break the UI; it shows reference/workflow options and a safe status message.
- API keys are masked and never returned raw.
- Unit tests cover provider selection, workflow patching, ComfyUI enqueue/history/view behavior, and route integration.
- Real ComfyUI smoke tests are opt-in.

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
