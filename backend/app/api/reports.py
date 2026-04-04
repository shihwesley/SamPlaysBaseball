"""Scouting report generation endpoints."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from backend.app.api.deps import get_storage
from backend.app.models.storage import StorageLayer
from backend.app.reports.generator import ReportGenerator, ScoutingReport

router = APIRouter(prefix="/api/reports", tags=["reports"])


def _get_generator(storage: StorageLayer) -> ReportGenerator:
    """Build ReportGenerator. LLM disabled unless ANTHROPIC_API_KEY is set."""
    import os

    llm = None
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            from backend.app.reports.llm import LLMReportGenerator

            llm = LLMReportGenerator(api_key=api_key)
        except Exception:
            pass
    return ReportGenerator(storage=storage, llm=llm)


@router.get("/{pitcher_id}", response_model=ScoutingReport)
def get_report(
    pitcher_id: str,
    report_type: str = Query(default="pitcher", description="pitcher|outing|pitch_type"),
    outing_date: str | None = Query(default=None, description="YYYY-MM-DD for outing reports"),
    pitch_type: str | None = Query(default=None, description="Pitch type for pitch_type reports"),
    format: str | None = Query(default=None, description="Set 'pdf' to download PDF"),
    storage: StorageLayer = Depends(get_storage),
):
    """Generate and return a scouting report. Add ?format=pdf to download as PDF."""
    gen = _get_generator(storage)

    if report_type == "outing":
        if not outing_date:
            raise HTTPException(status_code=400, detail="outing_date required for outing reports")
        report = gen.generate_outing_report(pitcher_id, outing_date)
    elif report_type == "pitch_type":
        if not pitch_type:
            raise HTTPException(status_code=400, detail="pitch_type required for pitch_type reports")
        report = gen.generate_pitch_type_report(pitcher_id, pitch_type)
    else:
        report = gen.generate_pitcher_report(pitcher_id)

    if format == "pdf":
        return _serve_pdf(report)

    return report


@router.get("/{pitcher_id}/pdf")
def get_report_pdf(
    pitcher_id: str,
    report_type: str = Query(default="pitcher"),
    outing_date: str | None = Query(default=None),
    pitch_type: str | None = Query(default=None),
    storage: StorageLayer = Depends(get_storage),
):
    """Download scouting report as PDF."""
    gen = _get_generator(storage)

    if report_type == "outing":
        if not outing_date:
            raise HTTPException(status_code=400, detail="outing_date required")
        report = gen.generate_outing_report(pitcher_id, outing_date)
    elif report_type == "pitch_type":
        if not pitch_type:
            raise HTTPException(status_code=400, detail="pitch_type required")
        report = gen.generate_pitch_type_report(pitcher_id, pitch_type)
    else:
        report = gen.generate_pitcher_report(pitcher_id)

    return _serve_pdf(report)


def _serve_pdf(report: ScoutingReport) -> FileResponse:
    from backend.app.reports.pdf import export_pdf

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.close()
    export_pdf(report, tmp.name)
    filename = f"{report.pitcher_id}_{report.report_type}_report.pdf"
    return FileResponse(
        path=tmp.name,
        media_type="application/pdf",
        filename=filename,
    )
