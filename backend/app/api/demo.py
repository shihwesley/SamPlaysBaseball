"""Demo mode endpoints — mirror real endpoints using ./data/demo/ storage."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from backend.app.api.models import (
    PitcherProfile,
    PitcherSummary,
    PitchListItem,
    PitchResponse,
    MeshFrame,
    MeshJoint,
    MeshResponse,
)
from backend.app.models.storage import StorageLayer

router = APIRouter(prefix="/api/demo", tags=["demo"])

_DEMO_DB = Path("./data/demo/baseball.db")
_DEMO_PARQUET = Path("./data/demo/parquet")


def _get_demo_storage() -> StorageLayer:
    if not _DEMO_DB.parent.exists():
        raise HTTPException(
            status_code=503,
            detail="Demo data directory ./data/demo/ not found. Load demo data first.",
        )
    return StorageLayer(db_path=_DEMO_DB, parquet_dir=_DEMO_PARQUET)


@router.get("/pitchers", response_model=list[PitcherSummary])
def demo_list_pitchers() -> list[PitcherSummary]:
    storage = _get_demo_storage()
    with storage._conn() as conn:
        rows = conn.execute(
            "SELECT pitcher_id, name, handedness FROM pitchers"
        ).fetchall()
    summaries = []
    for row in rows:
        pitch_ids = storage.list_pitch_ids(row["pitcher_id"])
        with storage._conn() as conn:
            type_rows = conn.execute(
                "SELECT DISTINCT pitch_type FROM pitches WHERE pitcher_id = ?",
                (row["pitcher_id"],),
            ).fetchall()
        pitch_types = [r["pitch_type"] for r in type_rows if r["pitch_type"]]
        summaries.append(
            PitcherSummary(
                pitcher_id=row["pitcher_id"],
                name=row["name"],
                handedness=row["handedness"],
                pitch_count=len(pitch_ids),
                pitch_types=pitch_types,
            )
        )
    return summaries


@router.get("/pitchers/{pitcher_id}", response_model=PitcherProfile)
def demo_get_pitcher(pitcher_id: str) -> PitcherProfile:
    storage = _get_demo_storage()
    baseline = storage.load_baseline(pitcher_id)
    if baseline is None:
        with storage._conn() as conn:
            row = conn.execute(
                "SELECT pitcher_id, name, handedness FROM pitchers WHERE pitcher_id = ?",
                (pitcher_id,),
            ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Demo pitcher {pitcher_id!r} not found")
        return PitcherProfile(pitcher_id=pitcher_id, name=row["name"], handedness=row["handedness"])
    sample_counts = {pt: bl.sample_count for pt, bl in baseline.by_pitch_type.items()}
    return PitcherProfile(
        pitcher_id=pitcher_id,
        name=baseline.pitcher_name,
        handedness=baseline.handedness,
        pitch_types=list(baseline.by_pitch_type.keys()),
        sample_counts=sample_counts,
    )


@router.get("/pitches/{pitch_id}", response_model=PitchResponse)
def demo_get_pitch(pitch_id: str) -> PitchResponse:
    storage = _get_demo_storage()
    pitch = storage.load_pitch(pitch_id)
    if pitch is None:
        raise HTTPException(status_code=404, detail=f"Demo pitch {pitch_id!r} not found")
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


@router.get("/pitches/{pitch_id}/mesh", response_model=MeshResponse)
def demo_get_pitch_mesh(pitch_id: str) -> MeshResponse:
    storage = _get_demo_storage()
    pitch = storage.load_pitch(pitch_id)
    if pitch is None:
        raise HTTPException(status_code=404, detail=f"Demo pitch {pitch_id!r} not found")
    raw = pitch.joints_mhr70 if pitch.joints_mhr70 is not None else pitch.joints
    frames = [
        MeshFrame(joints=[MeshJoint(x=j[0], y=j[1], z=j[2]) for j in frame])
        for frame in raw
    ]
    return MeshResponse(pitch_id=pitch_id, frames=frames)
