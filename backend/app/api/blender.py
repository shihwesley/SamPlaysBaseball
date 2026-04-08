"""Launch the local Blender viewer for a pitch's .npz mesh.

Local dev convenience endpoint — fires up Blender as a detached subprocess
with the field viewer script and the target NPZ injected via env var.
Assumes the backend is running on the same machine as the analyst (Mac dev).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.data.pitch_db import PitchDB

router = APIRouter(prefix="/api/blender", tags=["blender"])

PROJECT_ROOT = Path(__file__).resolve().parents[3]
VIEWER_SCRIPT = PROJECT_ROOT / "scripts" / "blender_field_viewer.py"

# Resolution order for the Blender executable.
_BLENDER_CANDIDATES = (
    os.environ.get("BLENDER_BIN") or "",
    "/Applications/Blender.app/Contents/MacOS/Blender",
    shutil.which("blender") or "",
)


def _resolve_blender() -> str:
    for c in _BLENDER_CANDIDATES:
        if c and Path(c).exists():
            return c
    raise HTTPException(
        status_code=503,
        detail=(
            "Blender executable not found. Install Blender, or set the "
            "BLENDER_BIN env var to its absolute path."
        ),
    )


class OpenRequest(BaseModel):
    play_id: str = Field(..., description="Pitch play_id whose .npz mesh to open")


class OpenResponse(BaseModel):
    status: str
    mesh_path: str
    pid: int
    blender_bin: str


@router.post("/open", response_model=OpenResponse)
def open_in_blender(req: OpenRequest):
    """Launch Blender with the pitcher's .npz loaded into the field viewer."""
    db = PitchDB()
    pitch = db.get_pitch(req.play_id)
    if pitch is None:
        raise HTTPException(status_code=404, detail=f"Unknown play_id: {req.play_id}")

    mesh_path_str = pitch.get("mesh_path")
    if not mesh_path_str:
        raise HTTPException(
            status_code=404,
            detail=f"Pitch {req.play_id} has no mesh on disk yet.",
        )

    mesh_path = Path(mesh_path_str)
    if not mesh_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Mesh file missing from disk: {mesh_path}",
        )

    if not VIEWER_SCRIPT.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Viewer script missing: {VIEWER_SCRIPT}",
        )

    blender_bin = _resolve_blender()

    env = os.environ.copy()
    env["SAMPLAYS_MESH_PATH"] = str(mesh_path)

    # Detached, non-blocking — Blender opens its own window and lives on its own.
    proc = subprocess.Popen(
        [blender_bin, "--python", str(VIEWER_SCRIPT)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    return OpenResponse(
        status="launched",
        mesh_path=str(mesh_path),
        pid=proc.pid,
        blender_bin=blender_bin,
    )
