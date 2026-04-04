"""Analysis result endpoints — reads stored results from StorageLayer."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend.app.api.deps import get_storage
from backend.app.api.models import BaselineSummary
from backend.app.models.storage import StorageLayer

router = APIRouter(prefix="/api/analysis", tags=["analysis"])


def _load_by_module(pitcher_id: str, module: str, storage: StorageLayer) -> list[dict]:
    """Load all analysis results for a pitcher filtered by module (single query)."""
    return storage.load_analysis_by_pitcher(pitcher_id, module=module)


@router.get("/tipping/{pitcher_id}")
def get_tipping(
    pitcher_id: str, storage: StorageLayer = Depends(get_storage)
) -> list[dict]:
    """All tipping detection results for a pitcher."""
    return _load_by_module(pitcher_id, "tipping-detection", storage)


@router.get("/fatigue/{pitcher_id}/{outing_id}")
def get_fatigue(
    pitcher_id: str,
    outing_id: str,
    storage: StorageLayer = Depends(get_storage),
) -> list[dict]:
    """Fatigue results for a specific outing date (YYYY-MM-DD)."""
    with storage._conn() as conn:
        rows = conn.execute(
            "SELECT pitch_id FROM pitches WHERE pitcher_id = ? AND game_date LIKE ?",
            (pitcher_id, f"{outing_id}%"),
        ).fetchall()
    pitch_ids = [r["pitch_id"] for r in rows]
    results: list[dict] = []
    for pid in pitch_ids:
        results.extend(storage.load_analysis(pid, module="fatigue-tracking"))
    return results


@router.get("/command/{pitcher_id}")
def get_command(
    pitcher_id: str, storage: StorageLayer = Depends(get_storage)
) -> list[dict]:
    return _load_by_module(pitcher_id, "command-analysis", storage)


@router.get("/arm-slot/{pitcher_id}")
def get_arm_slot(
    pitcher_id: str, storage: StorageLayer = Depends(get_storage)
) -> list[dict]:
    return _load_by_module(pitcher_id, "arm-slot-drift", storage)


@router.get("/timing/{pitch_id}")
def get_timing(
    pitch_id: str, storage: StorageLayer = Depends(get_storage)
) -> list[dict]:
    return storage.load_analysis(pitch_id, module="timing-analysis")


@router.get("/baseline/{pitcher_id}/{pitch_type}", response_model=BaselineSummary)
def get_baseline(
    pitcher_id: str,
    pitch_type: str,
    storage: StorageLayer = Depends(get_storage),
) -> BaselineSummary:
    """Baseline summary (no raw joint arrays) for a pitcher + pitch type."""
    baseline = storage.load_baseline(pitcher_id)
    if baseline is None:
        raise HTTPException(status_code=404, detail=f"No baseline for pitcher {pitcher_id!r}")
    pt_baseline = baseline.get_baseline(pitch_type)
    if pt_baseline is None:
        raise HTTPException(
            status_code=404,
            detail=f"No {pitch_type!r} baseline for pitcher {pitcher_id!r}",
        )
    return BaselineSummary(
        pitcher_id=pitcher_id,
        pitch_type=pitch_type,
        sample_count=pt_baseline.sample_count,
    )


@router.get("/injury-risk/{pitcher_id}")
def get_injury_risk(
    pitcher_id: str, storage: StorageLayer = Depends(get_storage)
) -> dict:
    """Most recent injury risk data for a pitcher."""
    results = _load_by_module(pitcher_id, "injury-risk", storage)
    if not results:
        return {"pitcher_id": pitcher_id, "risk_data": None, "message": "No injury risk data available"}
    # Return most recent (last in list)
    return {"pitcher_id": pitcher_id, "risk_data": results[-1]}
