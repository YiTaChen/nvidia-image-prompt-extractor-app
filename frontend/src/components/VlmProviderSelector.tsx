import { KeyRound, RefreshCw, Server } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import {
  getVlmProviders,
  listVlmModels,
  type VlmModelInfo,
  type VlmProviderInfo,
  type VlmSelection
} from "../api/client";

type Props = {
  value: VlmSelection;
  onChange: (value: VlmSelection) => void;
};

const FALLBACK_PROVIDERS: VlmProviderInfo[] = [
  {
    id: "nvidia",
    display_name: "NVIDIA",
    default_base_url: "https://integrate.api.nvidia.com/v1",
    default_model: "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
    requires_api_key: true,
    api_key_configured: false,
    supports_custom_base_url: true
  },
  {
    id: "lm_studio",
    display_name: "LM Studio",
    default_base_url: "http://localhost:1234/v1",
    default_model: "llava-v1.6",
    requires_api_key: false,
    api_key_configured: false,
    supports_custom_base_url: true
  },
  {
    id: "ollama",
    display_name: "Ollama",
    default_base_url: "http://localhost:11434",
    default_model: "llava:latest",
    requires_api_key: false,
    api_key_configured: false,
    supports_custom_base_url: true
  }
];

export function VlmProviderSelector({value, onChange}: Props) {
  const [providers, setProviders] = useState<VlmProviderInfo[]>([]);
  const [models, setModels] = useState<VlmModelInfo[]>([]);
  const [status, setStatus] = useState("");
  const [isLoadingProviders, setIsLoadingProviders] = useState(false);
  const [isLoadingModels, setIsLoadingModels] = useState(false);

  const providerOptions = providers.length ? providers : FALLBACK_PROVIDERS;
  const selectedProvider = useMemo(
    () => providerOptions.find((provider) => provider.id === value.provider) ?? providerOptions[0],
    [providerOptions, value.provider]
  );

  useEffect(() => {
    let isMounted = true;
    setIsLoadingProviders(true);
    getVlmProviders()
      .then((result) => {
        if (!isMounted) {
          return;
        }
        setProviders(result.providers.length ? result.providers : FALLBACK_PROVIDERS);
      })
      .catch(() => {
        if (isMounted) {
          setProviders(FALLBACK_PROVIDERS);
        }
      })
      .finally(() => {
        if (isMounted) {
          setIsLoadingProviders(false);
        }
      });
    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    if (!providers.length || !selectedProvider) {
      return;
    }
    if (!value.baseUrl || !value.model) {
      onChange({
        ...value,
        baseUrl: value.baseUrl || selectedProvider.default_base_url,
        model: value.model || selectedProvider.default_model
      });
    }
  }, [providers.length, selectedProvider, value, onChange]);

  useEffect(() => {
    if (!providers.length) {
      return;
    }
    void refreshModels();
  }, [providers.length, value.provider]);

  async function refreshModels() {
    setIsLoadingModels(true);
    setStatus("");
    try {
      const result = await listVlmModels(value);
      setModels(result.models);
      setStatus(result.message);
      const selectedExists = result.models.some((model) => model.id === value.model);
      const firstAvailable = result.models.find((model) => model.available) ?? result.models[0];
      if (!selectedExists && firstAvailable) {
        onChange({...value, model: firstAvailable.id});
      }
    } catch (err) {
      setModels([]);
      setStatus(err instanceof Error ? err.message : "模型列表讀取失敗。");
    } finally {
      setIsLoadingModels(false);
    }
  }

  function updateProvider(providerId: string) {
    const nextProvider = providerOptions.find((provider) => provider.id === providerId) ?? FALLBACK_PROVIDERS[0];
    onChange({
      provider: nextProvider.id,
      baseUrl: nextProvider.default_base_url,
      model: nextProvider.default_model,
      apiKey: ""
    });
    setModels([]);
    setStatus("");
  }

  const selectedModelVisible = models.some((model) => model.id === value.model);

  return (
    <div className="vlm-selector" aria-label="VLM 模型設定">
      <div className="vlm-field">
        <label htmlFor="vlm-provider">平台</label>
        <select id="vlm-provider" value={value.provider} onChange={(event) => updateProvider(event.target.value)}>
          {providerOptions.map((provider) => (
            <option key={provider.id} value={provider.id}>
              {provider.display_name}
            </option>
          ))}
        </select>
      </div>

      <div className="vlm-field">
        <label htmlFor="vlm-base-url">
          <Server aria-hidden="true" />
          URL
        </label>
        <input
          id="vlm-base-url"
          type="url"
          value={value.baseUrl}
          disabled={!selectedProvider?.supports_custom_base_url}
          onChange={(event) => onChange({...value, baseUrl: event.target.value})}
        />
      </div>

      <div className="vlm-field">
        <label htmlFor="vlm-api-key">
          <KeyRound aria-hidden="true" />
          Key
        </label>
        <input
          id="vlm-api-key"
          type="password"
          autoComplete="off"
          placeholder={selectedProvider?.api_key_configured ? "使用 .env" : ""}
          value={value.apiKey}
          onChange={(event) => onChange({...value, apiKey: event.target.value})}
        />
      </div>

      <div className="vlm-field vlm-model-field">
        <label htmlFor="vlm-model">模型</label>
        <div className="vlm-model-row">
          <select id="vlm-model" value={value.model} onChange={(event) => onChange({...value, model: event.target.value})}>
            {!selectedModelVisible && value.model ? <option value={value.model}>{value.model}</option> : null}
            {models.length ? (
              models.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.available ? "可用" : "參考"} · {model.display_name}
                </option>
              ))
            ) : (
              <option value={value.model}>{value.model || selectedProvider?.default_model || "選擇模型"}</option>
            )}
          </select>
          <button
            className="icon-button"
            type="button"
            aria-label="更新模型列表"
            title="更新模型列表"
            onClick={() => void refreshModels()}
            disabled={isLoadingProviders || isLoadingModels}
          >
            <RefreshCw className={isLoadingModels ? "spin" : ""} aria-hidden="true" />
          </button>
        </div>
      </div>

      {status ? <p className="vlm-status">{status}</p> : null}
    </div>
  );
}
