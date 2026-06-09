from pathlib import Path
from time import sleep

from fastapi.testclient import TestClient

from app.core.config import PROJECT_ROOT, get_settings
from app.core.job_manager import JobManager
from app.main import app


FIXTURE = Path(__file__).parent / "fixtures" / "sample.png"


def _configure_mock_settings():
    get_settings.cache_clear()
    settings = get_settings()
    settings.use_mock_nvidia = True
    settings.max_iterations = 3
    settings.image_output_size = 256
    return settings


def _wait_for_terminal(client: TestClient, job_id: str) -> dict:
    for _ in range(80):
        response = client.get(f"/api/jobs/{job_id}")
        payload = response.json()
        if payload["status"] in {"completed", "failed", "cancelled"}:
            return payload
        sleep(0.05)
    raise AssertionError("Job did not reach terminal status")


def test_job_api_runs_background_job_and_saves_images():
    _configure_mock_settings()
    client = TestClient(app)

    with FIXTURE.open("rb") as image:
        created = client.post(
            "/api/jobs",
            files={"image": ("sample.png", image, "image/png")},
            data={"threshold": "0", "max_iterations": "3"},
        )

    assert created.status_code == 200
    job_id = created.json()["job_id"]
    status = _wait_for_terminal(client, job_id)

    result = client.get(f"/api/jobs/{job_id}/result")
    result_payload = result.json()
    generated_path = PROJECT_ROOT / result_payload["attempts"][0]["generated_image_path"]

    assert status["status"] == "completed"
    assert status["original_image_path"] == f"backend/app/storage/jobs/{job_id}/original.png"
    assert result.status_code == 200
    assert result_payload["attempts"][0]["generated_image_path"].endswith("/generated/001.png")
    assert generated_path.exists()


def test_job_events_stream_returns_sse_events_for_completed_job():
    _configure_mock_settings()
    client = TestClient(app)

    with FIXTURE.open("rb") as image:
        created = client.post(
            "/api/jobs",
            files={"image": ("sample.png", image, "image/png")},
            data={"threshold": "0", "max_iterations": "3"},
        )

    job_id = created.json()["job_id"]
    _wait_for_terminal(client, job_id)

    response = client.get(f"/api/jobs/{job_id}/events")

    assert response.status_code == 200
    assert "data:" in response.text
    assert "completed" in response.text


def test_job_manager_can_cancel_queued_job(monkeypatch):
    _configure_mock_settings()
    manager = JobManager()
    monkeypatch.setattr(manager, "_ensure_worker", lambda: None)

    record = manager.create_job(FIXTURE.read_bytes(), threshold=80, max_iterations=3)
    cancelled = manager.cancel_job(record.job_id)

    assert cancelled is not None
    assert cancelled.status == "cancelled"
    assert cancelled.events[-1].type == "cancelled"


def test_cancel_unknown_job_returns_404():
    client = TestClient(app)

    response = client.post("/api/jobs/missing/cancel")

    assert response.status_code == 404
