import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ImageGenerationSelection } from "../api/client";
import { ImageGenerationProviderSelector } from "../components/ImageGenerationProviderSelector";

function TestHarness() {
  const [value, setValue] = useState<ImageGenerationSelection>({
    provider: "pollinations",
    baseUrl: "",
    apiKey: "",
    model: "kontext",
    workflow: ""
  });

  return <ImageGenerationProviderSelector value={value} onChange={setValue} />;
}

function jsonResponse(payload: unknown): Response {
  return {
    ok: true,
    json: () => Promise.resolve(payload)
  } as Response;
}

describe("ImageGenerationProviderSelector", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("shows ComfyUI text-to-image workflow settings after provider switch", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn((url: string) => {
        if (url.endsWith("/api/image-generation/providers")) {
          return Promise.resolve(
            jsonResponse({
              providers: [
                {
                  id: "pollinations",
                  display_name: "Pollinations",
                  default_base_url: "",
                  default_model: "kontext",
                  default_workflow: "",
                  requires_api_key: true,
                  api_key_configured: false,
                  supports_custom_base_url: false,
                  supports_workflows: false
                },
                {
                  id: "comfyui",
                  display_name: "ComfyUI",
                  default_base_url: "http://127.0.0.1:8188",
                  default_model: "",
                  default_workflow: "qwen_image_edit_plus_text_to_image",
                  requires_api_key: false,
                  api_key_configured: false,
                  supports_custom_base_url: true,
                  supports_workflows: true
                }
              ]
            })
          );
        }
        if (url.endsWith("/api/image-generation/workflows")) {
          return Promise.resolve(
            jsonResponse({
              workflows: [
                {
                  id: "qwen_image_edit_plus_text_to_image",
                  display_name: "Qwen Image Edit Plus - Text to Image",
                  mode: "text_to_image",
                  description: "",
                  workflow_path: "backend/app/workflows/comfyui/qwen_image_edit_plus_text_to_image.workflow.json",
                  required_checkpoint: "Qwen-Rapid-AIO-NSFW-v19.safetensors",
                  required_custom_nodes: ["TextEncodeQwenImageEditPlus"],
                  capabilities: ["text_to_image"],
                  primary: true
                }
              ]
            })
          );
        }
        return Promise.reject(new Error(`Unexpected request: ${url}`));
      })
    );

    render(<TestHarness />);
    const panel = screen.getByLabelText("生圖模型設定");

    await userEvent.selectOptions(within(panel).getByLabelText("平台"), "comfyui");

    await waitFor(() => expect(within(panel).getByLabelText("URL")).toHaveValue("http://127.0.0.1:8188"));
    expect(within(panel).getByLabelText("Checkpoint")).toBeInTheDocument();
    expect(within(panel).getByLabelText("Workflow")).toHaveValue("qwen_image_edit_plus_text_to_image");
  });
});
