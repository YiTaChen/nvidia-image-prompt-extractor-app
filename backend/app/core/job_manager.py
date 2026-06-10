from __future__ import annotations

import queue
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.core.config import get_settings
from app.core.image_io import bytes_to_data_url, load_normalized_image
from app.core.local_storage import path_for_response, save_generated_image, save_original_image
from app.core.refinement_loop import _best_attempt, _image_from_generation_result, _prompt_result_from_attempt, _refine_prompt
from app.core.services import get_image_generation_client, get_vision_client
from app.core.similarity import compare_images
from app.models.schemas import (
    ImageGenerationRequest,
    JobEvent,
    JobStatusResponse,
    RefinementAttempt,
    RefinementResult,
)


TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


@dataclass
class JobRecord:
    job_id: str
    original_content: bytes
    threshold: float
    max_iterations: int
    status: str = "queued"
    current_iteration: int = 0
    best_score: float | None = None
    original_image_path: str | None = None
    result: RefinementResult | None = None
    error: str | None = None
    cancel_requested: bool = False
    events: list[JobEvent] = field(default_factory=list)


class JobManager:
    def __init__(self):
        self._jobs: dict[str, JobRecord] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._lock = threading.RLock()
        self._worker: threading.Thread | None = None

    def create_job(self, original_content: bytes, threshold: float, max_iterations: int) -> JobRecord:
        settings = get_settings()
        job_id = uuid.uuid4().hex
        capped_iterations = max(1, min(settings.capped_max_iterations, max_iterations))
        original_path = save_original_image(job_id, original_content)
        record = JobRecord(
            job_id=job_id,
            original_content=original_content,
            threshold=max(0, min(100, threshold)),
            max_iterations=capped_iterations,
            original_image_path=path_for_response(original_path),
        )
        with self._lock:
            self._jobs[job_id] = record
            self._append_event(record, "queued", "Job queued.")
            self._queue.put(job_id)
            self._ensure_worker()
        return record

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel_job(self, job_id: str) -> JobRecord | None:
        with self._lock:
            record = self._jobs.get(job_id)
            if not record:
                return None
            if record.status in TERMINAL_STATUSES:
                return record
            record.cancel_requested = True
            if record.status == "queued":
                record.status = "cancelled"
                self._append_event(record, "cancelled", "Job cancelled before running.")
            else:
                self._append_event(record, "cancel_requested", "Cancellation requested.")
            return record

    def status_response(self, record: JobRecord) -> JobStatusResponse:
        return JobStatusResponse(
            job_id=record.job_id,
            status=record.status,
            threshold=record.threshold,
            max_iterations=record.max_iterations,
            current_iteration=record.current_iteration,
            best_score=record.best_score,
            original_image_path=record.original_image_path,
            error=record.error,
        )

    def events_since(self, job_id: str, cursor: int) -> tuple[list[JobEvent], int, bool]:
        with self._lock:
            record = self._jobs[job_id]
            events = record.events[cursor:]
            next_cursor = cursor + len(events)
            is_terminal = record.status in TERMINAL_STATUSES
            return events, next_cursor, is_terminal

    def _ensure_worker(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self._worker_loop, name="job-worker", daemon=True)
        self._worker.start()

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                self._run_job(job_id)
            finally:
                self._queue.task_done()

    def _run_job(self, job_id: str) -> None:
        with self._lock:
            record = self._jobs[job_id]
            if record.cancel_requested:
                record.status = "cancelled"
                self._append_event(record, "cancelled", "Job cancelled.")
                return
            record.status = "running"
            self._append_event(record, "running", "Job started.")

        try:
            self._execute_refinement(record)
        except Exception as exc:
            with self._lock:
                record.status = "failed"
                record.error = str(exc)
                self._append_event(record, "failed", str(exc))

    def _execute_refinement(self, record: JobRecord) -> None:
        settings = get_settings()
        original_image = load_normalized_image(record.original_content)
        vision_client = get_vision_client(settings)
        image_client = get_image_generation_client(settings)
        prompt_result = vision_client.generate_initial_prompt(bytes_to_data_url(record.original_content))
        attempts: list[RefinementAttempt] = []

        for iteration in range(1, record.max_iterations + 1):
            with self._lock:
                if record.cancel_requested:
                    record.status = "cancelled"
                    self._append_event(record, "cancelled", "Job cancelled.")
                    return
                record.current_iteration = iteration
                self._append_event(record, "iteration_started", f"Iteration {iteration} started.", iteration)

            generated = image_client.generate_image(
                ImageGenerationRequest(
                    prompt=prompt_result.prompt,
                    negative_prompt=prompt_result.negative_prompt,
                    width=settings.image_output_size,
                    height=settings.image_output_size,
                )
            )
            generated_path = save_generated_image(record.job_id, iteration, generated.image_base64, generated.mime_type)
            generated_image = _image_from_generation_result(generated.image_base64)
            score = compare_images(original_image, generated_image)
            attempt = RefinementAttempt(
                iteration=iteration,
                prompt=prompt_result.prompt,
                negative_prompt=prompt_result.negative_prompt,
                generated_image_base64=generated.image_base64,
                generated_image_mime_type=generated.mime_type,
                generated_image_path=path_for_response(generated_path),
                score=score,
            )
            attempts.append(attempt)

            with self._lock:
                record.best_score = max(record.best_score or 0, score.final_score)
                self._append_event(
                    record,
                    "iteration_completed",
                    f"Iteration {iteration} completed.",
                    iteration,
                    score.final_score,
                )

            if score.final_score >= record.threshold:
                break
            best_attempt = _best_attempt(attempts)
            prompt_result = _refine_prompt(
                vision_client,
                record.original_content,
                _image_from_generation_result(best_attempt.generated_image_base64),
                _prompt_result_from_attempt(best_attempt),
                best_attempt.score,
            )

        best_attempt = max(attempts, key=lambda attempt: attempt.score.final_score)
        with self._lock:
            record.result = RefinementResult(
                reached_threshold=best_attempt.score.final_score >= record.threshold,
                threshold=record.threshold,
                max_iterations=record.max_iterations,
                best_score=best_attempt.score.final_score,
                final_prompt=best_attempt.prompt,
                attempts=attempts,
            )
            record.status = "completed"
            self._append_event(record, "completed", "Job completed.", score=best_attempt.score.final_score)

    def _append_event(
        self,
        record: JobRecord,
        event_type: str,
        message: str,
        iteration: int | None = None,
        score: float | None = None,
    ) -> None:
        record.events.append(
            JobEvent(type=event_type, job_id=record.job_id, message=message, iteration=iteration, score=score)
        )


_job_manager = JobManager()


def get_job_manager() -> JobManager:
    return _job_manager
