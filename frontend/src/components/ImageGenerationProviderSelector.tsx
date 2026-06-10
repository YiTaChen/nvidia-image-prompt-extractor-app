import { KeyRound, Server, Workflow } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import {
  getImageGenerationProviders,
  getImageGenerationWorkflows,
  type ImageGenerationSelection,
  type ImageProviderInfo,
  type ImageWorkflowInfo
} from "../api/client";

type Props = {
  value: ImageGenerationSelection;
  onChange: (value: ImageGenerationSelection) => void;
};

const FALLBACK_PROVIDERS: ImageProviderInfo[] = [
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
];

const FALLBACK_WORKFLOWS: ImageWorkflowInfo[] = [
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
];

export function ImageGenerationProviderSelector({value, onChange}: Props) {
  const [providers, setProviders] = useState<ImageProviderInfo[]>([]);
  const [workflows, setWorkflows] = useState<ImageWorkflowInfo[]>([]);

  const providerOptions = useMemo(() => (providers.length ? providers : FALLBACK_PROVIDERS), [providers]);
  const workflowOptions = useMemo(
    () => (workflows.length ? workflows : FALLBACK_WORKFLOWS).filter((workflow) => workflow.mode === "text_to_image"),
    [workflows]
  );
  const selectedProvider = useMemo(
    () => providerOptions.find((provider) => provider.id === value.provider) ?? providerOptions[0],
    [providerOptions, value.provider]
  );

  useEffect(() => {
    let isMounted = true;
    getImageGenerationProviders()
      .then((result) => {
        if (isMounted) {
          setProviders(result.providers.length ? result.providers : FALLBACK_PROVIDERS);
        }
      })
      .catch(() => {
        if (isMounted) {
          setProviders(FALLBACK_PROVIDERS);
        }
      });
    getImageGenerationWorkflows()
      .then((result) => {
        if (isMounted) {
          setWorkflows(result.workflows.length ? result.workflows : FALLBACK_WORKFLOWS);
        }
      })
      .catch(() => {
        if (isMounted) {
          setWorkflows(FALLBACK_WORKFLOWS);
        }
      });
    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    if (!selectedProvider) {
      return;
    }
    const nextWorkflow = selectedProvider.supports_workflows
      ? value.workflow ||
        selectedProvider.default_workflow ||
        workflowOptions.find((workflow) => workflow.primary)?.id ||
        workflowOptions[0]?.id ||
        ""
      : "";
    const nextValue = {
      ...value,
      baseUrl: value.baseUrl || selectedProvider.default_base_url,
      model: value.model || selectedProvider.default_model,
      workflow: nextWorkflow
    };
    if (
      nextValue.baseUrl !== value.baseUrl ||
      nextValue.model !== value.model ||
      nextValue.workflow !== value.workflow
    ) {
      onChange(nextValue);
    }
  }, [selectedProvider, workflowOptions, value, onChange]);

  function updateProvider(providerId: string) {
    const nextProvider = providerOptions.find((provider) => provider.id === providerId) ?? FALLBACK_PROVIDERS[0];
    const nextWorkflow = nextProvider.supports_workflows
      ? nextProvider.default_workflow ||
        workflowOptions.find((workflow) => workflow.primary)?.id ||
        workflowOptions[0]?.id ||
        ""
      : "";
    onChange({
      provider: nextProvider.id,
      baseUrl: nextProvider.default_base_url,
      model: nextProvider.default_model,
      workflow: nextWorkflow,
      apiKey: ""
    });
  }

  const isComfyUi = value.provider === "comfyui";

  return (
    <div className="vlm-selector" aria-label="生圖模型設定">
      <div className="vlm-field">
        <label htmlFor="image-provider">平台</label>
        <select id="image-provider" value={value.provider} onChange={(event) => updateProvider(event.target.value)}>
          {providerOptions.map((provider) => (
            <option key={provider.id} value={provider.id}>
              {provider.display_name}
            </option>
          ))}
        </select>
      </div>

      <div className="vlm-field">
        <label htmlFor="image-api-key">
          <KeyRound aria-hidden="true" />
          Key
        </label>
        <input
          id="image-api-key"
          type="password"
          autoComplete="off"
          placeholder={selectedProvider?.api_key_configured ? "使用 .env" : ""}
          value={value.apiKey}
          onChange={(event) => onChange({...value, apiKey: event.target.value})}
        />
      </div>

      {isComfyUi ? (
        <>
          <div className="vlm-field">
            <label htmlFor="image-base-url">
              <Server aria-hidden="true" />
              URL
            </label>
            <input
              id="image-base-url"
              type="url"
              value={value.baseUrl}
              onChange={(event) => onChange({...value, baseUrl: event.target.value})}
            />
          </div>

          <div className="vlm-field">
            <label htmlFor="image-model">Checkpoint</label>
            <input
              id="image-model"
              type="text"
              placeholder="使用 workflow 預設"
              value={value.model}
              onChange={(event) => onChange({...value, model: event.target.value})}
            />
          </div>

          <div className="vlm-field vlm-model-field">
            <label htmlFor="image-workflow">
              <Workflow aria-hidden="true" />
              Workflow
            </label>
            <select id="image-workflow" value={value.workflow} onChange={(event) => onChange({...value, workflow: event.target.value})}>
              {workflowOptions.map((workflow) => (
                <option key={workflow.id} value={workflow.id}>
                  {workflow.display_name}
                </option>
              ))}
            </select>
          </div>
        </>
      ) : (
        <div className="vlm-field vlm-model-field">
          <label htmlFor="pollinations-model">模型</label>
          <input
            id="pollinations-model"
            type="text"
            value={value.model}
            onChange={(event) => onChange({...value, model: event.target.value})}
          />
        </div>
      )}
    </div>
  );
}
