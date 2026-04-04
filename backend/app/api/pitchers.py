"""Pitcher-related endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app.api.deps import get_storage
from backend.app.api.models import PitcherProfile, PitcherSummary, PitchListItem
from backend.app.models.storage import StorageLayer

router = APIRouter(prefix="/api/pitchers", tags=["pitchers"])


def _build_summaries(storage: StorageLayer) -> list[PitcherSummary]:
    """Build PitcherSummary list from storage."""
    with storage._conn() as conn:
        rows = conn.execute(
            "SELECT pitcher_id, name, handedness FROM pitchers"
        ).fetchall()
    summaries = []
    for row in rows:
        pitch_ids = storage.list_pitch_ids(row["pitcher_id"])
        # Count pitch types
        pitch_types: list[str] = []
        if pitch_ids:
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


@router.get("", response_model=list[PitcherSummary])
def list_pitchers(storage: StorageLayer = Depends(get_storage)) -> list[PitcherSummary]:
    """List all pitchers with summary stats."""
    return _build_summaries(storage)


@router.get("/{pitcher_id}", response_model=PitcherProfile)
def get_pitcher(
    pitcher_id: str, storage: StorageLayer = Depends(get_storage)
) -> PitcherProfile:
    """Pitcher profile + baseline summary."""
    baseline = storage.load_baseline(pitcher_id)
    if baseline is None:
        # Check if pitcher exists at all
        with storage._conn() as conn:
            row = conn.execute(
                "SELECT pitcher_id, name, handedness FROM pitchers WHERE pitcher_id = ?",
                (pitcher_id,),
            ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Pitcher {pitcher_id!r} not found")
        return PitcherProfile(
            pitcher_id=pitcher_id,
            name=row["name"],
            handedness=row["handedness"],
        )
    sample_counts = {
        pt: bl.sample_count for pt, bl in baseline.by_pitch_type.items()
    }
    return PitcherProfile(
        pitcher_id=pitcher_id,
        name=baseline.pitcher_name,
        handedness=baseline.handedness,
        pitch_types=list(baseline.by_pitch_type.keys()),
        sample_counts=sample_counts,
    )


@router.get("/{pitcher_id}/pitches", response_model=list[PitchListItem])
def list_pitcher_pitches(
    pitcher_id: str,
    pitch_type: str | None = Query(default=None),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    storage: StorageLayer = Depends(get_storage),
) -> list[PitchListItem]:
    """Paginated, filterable pitch list for a pitcher."""
    query = "SELECT pitch_id, pitcher_id, game_date, pitch_type, velocity_mph, result FROM pitches WHERE pitcher_id = ?"
    params: list = [pitcher_id]
    if pitch_type:
        query += " AND pitch_type = ?"
        params.append(pitch_type)
    if date_from:
        query += " AND game_date >= ?"
        params.append(date_from)
    if date_to:
        query += " AND game_date <= ?"
        params.append(date_to)
    query += " ORDER BY game_date, pitch_number LIMIT ? OFFSET ?"
    params += [page_size, (page - 1) * page_size]
    with storage._conn() as conn:
        rows = conn.execute(query, params).fetchall()
    return [
        PitchListItem(
            pitch_id=r["pitch_id"],
            pitcher_id=r["pitcher_id"],
            game_date=r["game_date"],
            pitch_type=r["pitch_type"],
            velocity_mph=r["velocity_mph"],
            result=r["result"],
        )
        for r in rows
    ]
