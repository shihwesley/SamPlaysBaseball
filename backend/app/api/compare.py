"""Side-by-side pitch comparison endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app.api.deps import get_storage
from backend.app.api.models import CompareResponse, PitchResponse
from backend.app.models.storage import StorageLayer

router = APIRouter(tags=["compare"])


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


@router.get("/api/compare", response_model=CompareResponse)
def compare_pitches(
    pitch_a: str = Query(..., description="First pitch ID"),
    pitch_b: str = Query(..., description="Second pitch ID"),
    storage: StorageLayer = Depends(get_storage),
) -> CompareResponse:
    """Compare two pitches side by side."""
    return CompareResponse(
        pitch_a=_pitch_to_response(pitch_a, storage),
        pitch_b=_pitch_to_response(pitch_b, storage),
    )
