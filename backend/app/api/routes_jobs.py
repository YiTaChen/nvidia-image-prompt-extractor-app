import asyncio
import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.core.image_io import ImageValidationError
from app.core.job_manager import TERMINAL_STATUSES, get_job_manager
from app.models.schemas import JobCreateResponse, JobStatusResponse, RefinementResult


router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.post("", response_model=JobCreateResponse)
async def create_job(
    image: UploadFile = File(...),
    threshold: float = Form(default=80),
    max_iterations: int = Form(default=3),
) -> JobCreateResponse:
    content = await image.read()
    try:
        record = get_job_manager().create_job(content, threshold, max_iterations)
    except ImageValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Job creation failed: {exc}") from exc
    return JobCreateResponse(job_id=record.job_id, status=record.status)


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str) -> JobStatusResponse:
    record = get_job_manager().get_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found.")
    return get_job_manager().status_response(record)


@router.get("/{job_id}/result", response_model=RefinementResult)
async def get_job_result(job_id: str) -> RefinementResult:
    record = get_job_manager().get_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found.")
    if record.status != "completed" or not record.result:
        raise HTTPException(status_code=409, detail=f"Job is {record.status}.")
    return record.result


@router.post("/{job_id}/cancel", response_model=JobStatusResponse)
async def cancel_job(job_id: str) -> JobStatusResponse:
    record = get_job_manager().cancel_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found.")
    return get_job_manager().status_response(record)


@router.get("/{job_id}/events")
async def job_events(job_id: str) -> StreamingResponse:
    if not get_job_manager().get_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found.")

    async def stream():
        cursor = 0
        while True:
            events, cursor, is_terminal = get_job_manager().events_since(job_id, cursor)
            for event in events:
                yield f"data: {json.dumps(event.model_dump())}\n\n"
            if is_terminal:
                break
            await asyncio.sleep(0.25)

    return StreamingResponse(stream(), media_type="text/event-stream")
