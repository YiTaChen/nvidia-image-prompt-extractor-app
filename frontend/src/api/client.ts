export type PromptExtractionResult = {
  prompt: string;
  negative_prompt: string;
  analysis: Record<string, unknown>;
};

export type ImageGenerationResult = {
  image_base64: string;
  mime_type: string;
  model: string;
  provider?: string | null;
  workflow?: string | null;
  seed?: number | null;
  mode?: string | null;
  metadata?: Record<string, unknown>;
};

export type RefinementAttempt = {
  iteration: number;
  prompt: string;
  negative_prompt: string;
  generated_image_base64: string;
  generated_image_mime_type: string;
  score: {
    final_score: number;
    histogram_score: number;
    average_hash_score: number;
  };
};

export type RefinementResult = {
  reached_threshold: boolean;
  threshold: number;
  max_iterations: number;
  best_score: number;
  final_prompt: string;
  attempts: RefinementAttempt[];
};

export type JobCreateResult = {
  job_id: string;
  status: string;
};

export type JobEvent = {
  type: string;
  job_id: string;
  message: string;
  iteration: number | null;
  score: number | null;
};

export type VlmProviderInfo = {
  id: string;
  display_name: string;
  default_base_url: string;
  default_model: string;
  requires_api_key: boolean;
  api_key_configured: boolean;
  supports_custom_base_url: boolean;
};

export type VlmModelInfo = {
  id: string;
  display_name: string;
  provider: string;
  available: boolean;
  capabilities: string[];
  source: string;
  reason: string | null;
};

export type VlmProvidersResult = {
  providers: VlmProviderInfo[];
};

export type VlmModelListResult = {
  provider: string;
  connection_status: string;
  message: string;
  models: VlmModelInfo[];
};

export type VlmSelection = {
  provider: string;
  model: string;
  baseUrl: string;
  apiKey: string;
};

export type ImageProviderInfo = {
  id: string;
  display_name: string;
  default_base_url: string;
  default_model: string;
  default_workflow: string;
  requires_api_key: boolean;
  api_key_configured: boolean;
  supports_custom_base_url: boolean;
  supports_workflows: boolean;
};

export type ImageWorkflowInfo = {
  id: string;
  display_name: string;
  mode: string;
  description: string;
  workflow_path: string;
  required_checkpoint: string;
  required_custom_nodes: string[];
  capabilities: string[];
  primary: boolean;
};

export type ImageGenerationSelection = {
  provider: string;
  baseUrl: string;
  apiKey: string;
  model: string;
  workflow: string;
};

export async function getVlmProviders(): Promise<VlmProvidersResult> {
  const response = await fetch("/api/vlm/providers");
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return response.json();
}

export async function listVlmModels(selection: Partial<VlmSelection> & {provider: string}): Promise<VlmModelListResult> {
  const response = await fetch("/api/vlm/models", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      provider: selection.provider,
      base_url: selection.baseUrl || null,
      api_key: selection.apiKey || null
    })
  });
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return response.json();
}

export async function extractPrompt(image: File, vlm?: VlmSelection): Promise<PromptExtractionResult> {
  const formData = new FormData();
  formData.append("image", image);
  if (vlm?.provider) {
    formData.append("vlm_provider", vlm.provider);
  }
  if (vlm?.model) {
    formData.append("vlm_model", vlm.model);
  }
  if (vlm?.baseUrl) {
    formData.append("vlm_base_url", vlm.baseUrl);
  }
  if (vlm?.apiKey) {
    formData.append("vlm_api_key", vlm.apiKey);
  }
  const response = await fetch("/api/extract-prompt", {
    method: "POST",
    body: formData
  });
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return response.json();
}

export async function getImageGenerationProviders(): Promise<{providers: ImageProviderInfo[]}> {
  const response = await fetch("/api/image-generation/providers");
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return response.json();
}

export async function getImageGenerationWorkflows(): Promise<{workflows: ImageWorkflowInfo[]}> {
  const response = await fetch("/api/image-generation/workflows");
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return response.json();
}

export async function generateImage(
  prompt: string,
  negativePrompt = "",
  selection?: ImageGenerationSelection
): Promise<ImageGenerationResult> {
  const response = await fetch("/api/generate-image", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      prompt,
      negative_prompt: negativePrompt,
      width: 1024,
      height: 1024,
      image_provider: selection?.provider || null,
      image_base_url: selection?.baseUrl || null,
      image_api_key: selection?.apiKey || null,
      image_model: selection?.model || null,
      image_workflow: selection?.workflow || null
    })
  });
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return response.json();
}

export async function runRefinementLoop(image: File, threshold: number, maxIterations: number): Promise<RefinementResult> {
  const formData = new FormData();
  formData.append("image", image);
  formData.append("threshold", String(threshold));
  formData.append("max_iterations", String(maxIterations));
  const response = await fetch("/api/refine-image", {
    method: "POST",
    body: formData
  });
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return response.json();
}

export async function createRefinementJob(image: File, threshold: number, maxIterations: number): Promise<JobCreateResult> {
  const formData = new FormData();
  formData.append("image", image);
  formData.append("threshold", String(threshold));
  formData.append("max_iterations", String(maxIterations));
  const response = await fetch("/api/jobs", {
    method: "POST",
    body: formData
  });
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return response.json();
}

export async function getJobResult(jobId: string): Promise<RefinementResult> {
  const response = await fetch(`/api/jobs/${jobId}/result`);
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return response.json();
}

export async function cancelJob(jobId: string): Promise<void> {
  const response = await fetch(`/api/jobs/${jobId}/cancel`, {
    method: "POST"
  });
  if (!response.ok) {
    throw new Error(await readError(response));
  }
}

async function readError(response: Response): Promise<string> {
  try {
    const data = await response.json();
    return data.detail ?? response.statusText;
  } catch {
    return response.statusText;
  }
}
