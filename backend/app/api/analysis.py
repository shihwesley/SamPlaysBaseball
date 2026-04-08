"""Analysis result endpoints — reads stored results from StorageLayer.

Note: As of the 2026-04-07 strategic pivot, the `/fatigue/`, `/command/`, and
`/injury-risk/` endpoints have been deprecated and now return HTTP 410 Gone.
The underlying analysis modules in `backend/app/analysis/` are preserved but
are no longer surfaced through the API or the report generator. See VALIDATION.md
"Future Biomechanics Work" for the rationale and revival path.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend.app.api.deps import get_storage
from backend.app.api.models import BaselineSummary
from backend.app.models.storage import StorageLayer

router = APIRouter(prefix="/api/analysis", tags=["analysis"])

_DEFERRED_DETAIL = (
    "This analysis module has been moved to Future Biomechanics Work. "
    "See VALIDATION.md for the validation gap and revival path."
)


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
def get_fatigue(pitcher_id: str, outing_id: str) -> dict:
    """Deferred — see VALIDATION.md → Future Biomechanics Work."""
    raise HTTPException(status_code=410, detail=_DEFERRED_DETAIL)


@router.get("/command/{pitcher_id}")
def get_command(pitcher_id: str) -> dict:
    """Deferred — see VALIDATION.md → Future Biomechanics Work."""
    raise HTTPException(status_code=410, detail=_DEFERRED_DETAIL)


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
def get_injury_risk(pitcher_id: str) -> dict:
    """Deferred — see VALIDATION.md → Future Biomechanics Work.

    Combining unvalidated mechanical signals into a medical-adjacent risk score
    is exactly the failure mode that makes team doctors reject biomechanics tools.
    The composite score has been removed from the user-facing flow until each input
    signal is validated against marker-mocap ground truth (Driveline OpenBiomechanics)
    and the composite is validated against published UCL injury cohort data.
    """
    raise HTTPException(status_code=410, detail=_DEFERRED_DETAIL)
