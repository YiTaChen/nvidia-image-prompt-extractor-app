import { ImageUp, Loader2, Repeat } from "lucide-react";
import { useState } from "react";

import {
  cancelJob,
  createRefinementJob,
  getJobResult,
  type JobEvent,
  type RefinementResult
} from "../api/client";

export function RefinementLoopPanel() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [threshold, setThreshold] = useState(80);
  const [maxIterations, setMaxIterations] = useState(3);
  const [result, setResult] = useState<RefinementResult | null>(null);
  const [events, setEvents] = useState<JobEvent[]>([]);
  const [jobId, setJobId] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  function handleFileChange(nextFile: File | null) {
    setFile(nextFile);
    setResult(null);
    setEvents([]);
    setJobId("");
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
    setEvents([]);
    setResult(null);
    try {
      const created = await createRefinementJob(file, threshold, maxIterations);
      setJobId(created.job_id);
      const eventSource = new EventSource(`/api/jobs/${created.job_id}/events`);
      eventSource.onmessage = async (event) => {
        const parsed = JSON.parse(event.data) as JobEvent;
        setEvents((current) => [...current, parsed]);
        if (["completed", "failed", "cancelled"].includes(parsed.type)) {
          eventSource.close();
          setIsLoading(false);
          if (parsed.type === "completed") {
            setResult(await getJobResult(created.job_id));
          } else {
            setError(parsed.message);
          }
        }
      };
      eventSource.onerror = () => {
        eventSource.close();
        setIsLoading(false);
        setError("事件連線中斷。");
      };
    } catch (err) {
      setError(err instanceof Error ? err.message : "Loop 執行失敗。");
      setIsLoading(false);
    }
  }

  async function handleCancel() {
    if (!jobId) {
      return;
    }
    try {
      await cancelJob(jobId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "取消失敗。");
    }
  }

  const bestAttempt = result?.attempts.reduce((best, attempt) =>
    attempt.score.final_score > best.score.final_score ? attempt : best
  );

  return (
    <section className="panel wide-panel">
      <div className="section-title">
        <Repeat aria-hidden="true" />
        <h2>Prompt Refinement Loop</h2>
      </div>

      <div className="loop-grid">
        <label className="upload-box compact-upload">
          <input
            aria-label="選擇 Loop 圖片"
            type="file"
            accept="image/png,image/jpeg,image/webp"
            onChange={(event) => handleFileChange(event.target.files?.[0] ?? null)}
          />
          {previewUrl ? <img src={previewUrl} alt="Loop 上傳預覽" /> : <span><ImageUp aria-hidden="true" /> 選擇圖片</span>}
        </label>

        <div className="loop-controls">
          <label htmlFor="loop-threshold">相似度門檻</label>
          <input
            id="loop-threshold"
            type="number"
            min="0"
            max="100"
            value={threshold}
            onChange={(event) => setThreshold(Number(event.target.value))}
          />

          <label htmlFor="loop-max-iterations">最高迭代次數</label>
          <input
            id="loop-max-iterations"
            type="number"
            min="1"
            max="3"
            value={maxIterations}
            onChange={(event) => setMaxIterations(Math.min(3, Math.max(1, Number(event.target.value))))}
          />

          <button className="primary-button" type="button" onClick={handleSubmit} disabled={isLoading}>
            {isLoading ? <Loader2 className="spin" aria-hidden="true" /> : <Repeat aria-hidden="true" />}
            執行 Loop
          </button>
          <button className="secondary-button" type="button" onClick={handleCancel} disabled={!isLoading || !jobId}>
            取消 Job
          </button>
        </div>
      </div>

      {error ? <p className="error">{error}</p> : null}
      {events.length ? (
        <div className="event-list">
          {events.map((event, index) => (
            <span key={`${event.type}-${index}`}>
              {event.iteration ? `#${event.iteration} ` : ""}
              {event.message}
              {event.score !== null ? ` (${event.score.toFixed(1)})` : ""}
            </span>
          ))}
        </div>
      ) : null}

      {result && bestAttempt ? (
        <div className="loop-result">
          <div>
            <strong>{result.reached_threshold ? "已達標" : "未達標"}</strong>
            <span>最佳分數 {result.best_score.toFixed(1)} / 門檻 {result.threshold}</span>
          </div>
          <img
            src={`data:${bestAttempt.generated_image_mime_type};base64,${bestAttempt.generated_image_base64}`}
            alt="Loop 最佳生成圖"
          />
          <textarea value={result.final_prompt} readOnly />
          <div className="attempt-list">
            {result.attempts.map((attempt) => (
              <span key={attempt.iteration}>
                #{attempt.iteration} {attempt.score.final_score.toFixed(1)}
              </span>
            ))}
          </div>
        </div>
      ) : null}
    </section>
  );
}
