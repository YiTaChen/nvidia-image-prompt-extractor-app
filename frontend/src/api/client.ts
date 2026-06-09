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

async function readError(response: Response): Promise<string> {
  try {
    const data = await response.json();
    return data.detail ?? response.statusText;
  } catch {
    return response.statusText;
  }
}
