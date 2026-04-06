"""Video upload and job management endpoints."""

from __future__ import annotations

import asyncio
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile

from backend.app.api.deps import get_job_store, get_storage
from backend.app.api.models import JobStatus, UploadResponse
from backend.app.models.storage import StorageLayer

router = APIRouter(tags=["upload"])

_VIDEO_DIR = Path("./data/videos")


async def _process_video_stub(job_id: str, job_store: dict[str, JobStatus]) -> None:
    """Stub background processor — simulates completion."""
    if job_id not in job_store:
        return
    job = job_store[job_id]
    job.status = "processing"
    job.progress = 10
    job.updated_at = datetime.now(tz=timezone.utc)
    await asyncio.sleep(0.1)
    job.status = "completed"
    job.progress = 100
    job.updated_at = datetime.now(tz=timezone.utc)


@router.post("/api/upload", response_model=UploadResponse, status_code=202)
async def upload_video(
    background_tasks: BackgroundTasks,
    pitcher_id: str = Form(...),
    game_date: str = Form(...),
    inning: int = Form(...),
    pitch_number: int = Form(...),
    pitch_type: str = Form(...),
    video: UploadFile = File(...),
    storage: StorageLayer = Depends(get_storage),
    job_store: dict = Depends(get_job_store),
) -> UploadResponse:
    """Accept video upload + metadata, queue background processing."""
    # Sanitize pitcher_id to prevent path traversal
    if not re.fullmatch(r"[A-Za-z0-9_\-]{1,64}", pitcher_id):
        raise HTTPException(
            status_code=400,
            detail="pitcher_id must be alphanumeric with hyphens/underscores only (max 64 chars)",
        )

    job_id = str(uuid.uuid4())
    now = datetime.now(tz=timezone.utc)

    # Save video file
    video_dir = _VIDEO_DIR / pitcher_id
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"{job_id}_{video.filename or 'video.mp4'}"
    content = await video.read()
    video_path.write_bytes(content)

    job = JobStatus(
        job_id=job_id,
        status="queued",
        progress=0,
        message=f"Video queued for processing: {video.filename}",
        created_at=now,
        updated_at=now,
    )
    job_store[job_id] = job

    background_tasks.add_task(_process_video_stub, job_id, job_store)

    return UploadResponse(job_id=job_id, status="queued")


@router.get("/api/jobs/{job_id}", response_model=JobStatus)
def get_job(
    job_id: str,
    job_store: dict = Depends(get_job_store),
) -> JobStatus:
    """Get processing job status."""
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return job
