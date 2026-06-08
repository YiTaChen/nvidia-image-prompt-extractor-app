import { Image, Loader2, Wand2 } from "lucide-react";
import { useEffect, useState } from "react";

import { generateImage, type PromptExtractionResult } from "../api/client";

type Props = {
  extractedPrompt: PromptExtractionResult | null;
};

export function PromptImagePanel({extractedPrompt}: Props) {
  const [prompt, setPrompt] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [imageUrl, setImageUrl] = useState("");
  const [model, setModel] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (extractedPrompt) {
      setPrompt(extractedPrompt.prompt);
      setNegativePrompt(extractedPrompt.negative_prompt);
    }
  }, [extractedPrompt]);

  async function handleSubmit() {
    setIsLoading(true);
    setError("");
    setImageUrl("");
    try {
      const result = await generateImage(prompt, negativePrompt);
      setImageUrl(`data:${result.mime_type};base64,${result.image_base64}`);
      setModel(result.model);
    } catch (err) {
      setError(err instanceof Error ? err.message : "生圖失敗。");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <section className="panel">
      <div className="section-title">
        <Image aria-hidden="true" />
        <h2>Prompt 生成圖片</h2>
      </div>

      <label htmlFor="manual-prompt">Prompt</label>
      <textarea
        id="manual-prompt"
        value={prompt}
        onChange={(event) => setPrompt(event.target.value)}
        placeholder="輸入或使用左側產生的 prompt"
      />

      <label htmlFor="manual-negative-prompt">Negative Prompt</label>
      <textarea
        id="manual-negative-prompt"
        value={negativePrompt}
        onChange={(event) => setNegativePrompt(event.target.value)}
        placeholder="不想出現的元素"
      />

      <button className="primary-button" type="button" onClick={handleSubmit} disabled={isLoading || !prompt.trim()}>
        {isLoading ? <Loader2 className="spin" aria-hidden="true" /> : <Wand2 aria-hidden="true" />}
        生成圖片
      </button>

      {error ? <p className="error">{error}</p> : null}
      {imageUrl ? (
        <div className="generated-preview">
          <img src={imageUrl} alt="生成圖片" />
          <span>{model}</span>
        </div>
      ) : null}
    </section>
  );
}
