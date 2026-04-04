"""WebSocket endpoint for real-time job progress."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.app.api.deps import get_job_store_from_app

router = APIRouter(tags=["websocket"])


@router.websocket("/api/ws/jobs/{job_id}")
async def job_progress_ws(websocket: WebSocket, job_id: str) -> None:
    """Stream job status updates until the job completes or fails."""
    await websocket.accept()
    job_store = get_job_store_from_app(websocket.app)
    try:
        while True:
            job = job_store.get(job_id)
            if job is None:
                await websocket.send_text(
                    json.dumps({"error": f"Job {job_id!r} not found"})
                )
                break
            payload = {
                "job_id": job.job_id,
                "status": job.status,
                "progress": job.progress,
                "message": job.message,
                "pitch_id": job.pitch_id,
            }
            await websocket.send_text(json.dumps(payload))
            if job.status in ("completed", "failed"):
                break
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    finally:
        await websocket.close()
