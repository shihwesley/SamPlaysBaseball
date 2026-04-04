"""Pitch data endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend.app.api.deps import get_storage
from backend.app.api.models import MeshFrame, MeshJoint, MeshResponse, PitchResponse
from backend.app.models.storage import StorageLayer

router = APIRouter(prefix="/api/pitches", tags=["pitches"])


def _pitch_to_response(pitch_id: str, storage: StorageLayer) -> PitchResponse:
    pitch = storage.load_pitch(pitch_id)
    if pitch is None:
        raise HTTPException(status_code=404, detail=f"Pitch {pitch_id!r} not found")
    analysis = storage.load_analysis(pitch_id)
    m = pitch.metadata
    return PitchResponse(
        pitch_id=m.pitch_id,
        pitcher_id=m.pitcher_id,
        game_date=m.game_date.isoformat(),
        pitch_type=m.pitch_type,
        velocity_mph=m.velocity_mph,
        spin_rate_rpm=m.spin_rate_rpm,
        plate_x=m.plate_x,
        plate_z=m.plate_z,
        result=m.result,
        num_frames=pitch.num_frames,
        joints=pitch.joints,
        joints_mhr70=pitch.joints_mhr70,
        pose_params=pitch.pose_params,
        shape_params=pitch.shape_params,
        analysis_results=analysis,
    )


@router.get("/{pitch_id}", response_model=PitchResponse)
def get_pitch(
    pitch_id: str, storage: StorageLayer = Depends(get_storage)
) -> PitchResponse:
    """Full pitch data + all analysis results."""
    return _pitch_to_response(pitch_id, storage)


@router.get("/{pitch_id}/mesh", response_model=MeshResponse)
def get_pitch_mesh(
    pitch_id: str, storage: StorageLayer = Depends(get_storage)
) -> MeshResponse:
    """MHR70 joint positions formatted for Three.js."""
    pitch = storage.load_pitch(pitch_id)
    if pitch is None:
        raise HTTPException(status_code=404, detail=f"Pitch {pitch_id!r} not found")
    if pitch.joints_mhr70 is None:
        # Fall back to full joints if mhr70 not available
        raw = pitch.joints
    else:
        raw = pitch.joints_mhr70
    frames = [
        MeshFrame(joints=[MeshJoint(x=j[0], y=j[1], z=j[2]) for j in frame])
        for frame in raw
    ]
    return MeshResponse(pitch_id=pitch_id, frames=frames)
