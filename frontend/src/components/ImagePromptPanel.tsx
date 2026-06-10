import { ImageUp, Loader2, Sparkles } from "lucide-react";
import { useMemo, useState } from "react";

import { extractPrompt, type PromptExtractionResult, type VlmSelection } from "../api/client";
import { VlmProviderSelector } from "./VlmProviderSelector";

type Props = {
  onPromptReady: (result: PromptExtractionResult) => void;
};

export function ImagePromptPanel({onPromptReady}: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState<PromptExtractionResult | null>(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [vlmSelection, setVlmSelection] = useState<VlmSelection>({
    provider: "nvidia",
    model: "",
    baseUrl: "",
    apiKey: ""
  });

  const fileName = useMemo(() => file?.name ?? "尚未選擇圖片", [file]);

  function handleFileChange(nextFile: File | null) {
    setFile(nextFile);
    setResult(null);
    setError("");
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(nextFile ? URL.createObjectURL(nextFile) : "");
  }

  async function handleSubmit() {
    if (!file) {
      setError("請先選擇圖片。");
      return;
    }
    setIsLoading(true);
    setError("");
    try {
      const nextResult = await extractPrompt(file, vlmSelection);
      setResult(nextResult);
      onPromptReady(nextResult);
    } catch (err) {
      setError(err instanceof Error ? err.message : "產生 prompt 失敗。");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <section className="panel">
      <div className="section-title">
        <ImageUp aria-hidden="true" />
        <h2>圖片生成 Prompt</h2>
      </div>

      <label className="upload-box">
        <input
          aria-label="選擇圖片"
          type="file"
          accept="image/png,image/jpeg,image/webp"
          onChange={(event) => handleFileChange(event.target.files?.[0] ?? null)}
        />
        {previewUrl ? <img src={previewUrl} alt="上傳預覽" /> : <span>{fileName}</span>}
      </label>

      <VlmProviderSelector value={vlmSelection} onChange={setVlmSelection} />

      <button className="primary-button" type="button" onClick={handleSubmit} disabled={isLoading}>
        {isLoading ? <Loader2 className="spin" aria-hidden="true" /> : <Sparkles aria-hidden="true" />}
        產生 Prompt
      </button>

      {error ? <p className="error">{error}</p> : null}
      {result ? (
        <div className="result-block">
          <label htmlFor="extracted-prompt">Prompt</label>
          <textarea id="extracted-prompt" value={result.prompt} readOnly />
          <label htmlFor="negative-prompt">Negative Prompt</label>
          <textarea id="negative-prompt" value={result.negative_prompt} readOnly />
        </div>
      ) : null}
    </section>
  );
}
