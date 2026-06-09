import { useState } from "react";

import type { PromptExtractionResult } from "./api/client";
import { ImagePromptPanel } from "./components/ImagePromptPanel";
import { PromptImagePanel } from "./components/PromptImagePanel";
import { RefinementLoopPanel } from "./components/RefinementLoopPanel";
import "./styles.css";

export default function App() {
  const [extractedPrompt, setExtractedPrompt] = useState<PromptExtractionResult | null>(null);

  return (
    <main>
      <header>
        <h1>NVIDIA 圖片 Prompt 提取 App</h1>
      </header>
      <div className="workspace">
        <ImagePromptPanel onPromptReady={setExtractedPrompt} />
        <PromptImagePanel extractedPrompt={extractedPrompt} />
        <RefinementLoopPanel />
      </div>
    </main>
  );
}
