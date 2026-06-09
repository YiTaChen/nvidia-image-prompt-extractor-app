export type PromptExtractionResult = {
  prompt: string;
  negative_prompt: string;
  analysis: Record<string, string>;
};

export type ImageGenerationResult = {
  image_base64: string;
  mime_type: string;
  model: string;
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

export async function extractPrompt(image: File): Promise<PromptExtractionResult> {
  const formData = new FormData();
  formData.append("image", image);
  const response = await fetch("/api/extract-prompt", {
    method: "POST",
    body: formData
  });
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return response.json();
}

export async function generateImage(prompt: string, negativePrompt = ""): Promise<ImageGenerationResult> {
  const response = await fetch("/api/generate-image", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      prompt,
      negative_prompt: negativePrompt,
      width: 1024,
      height: 1024
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
