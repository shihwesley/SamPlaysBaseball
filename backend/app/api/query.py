"""Query endpoints — POST /api/query, GET /api/query/{token}/status."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.app.api.deps import get_job_store

router = APIRouter(prefix="/api/query", tags=["query"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    text: str = Field(..., description="Natural language query from the analyst")
    game_date: str | None = Field(None, description="Optional game date override (YYYY-MM-DD)")


class ParsedQueryResponse(BaseModel):
    pitcher_name: str
    pitch_types: list[str]
    comparison_mode: str
    concern: str | None = None


class ReportResponse(BaseModel):
    narrative: str
    recommendations: list[str]
    risk_flags: list[str]
    confidence: str
    pitches_analyzed: int


class ViewerResponse(BaseModel):
    glb_url: str
    phase_markers: dict[str, int] = Field(default_factory=dict)
    total_frames: int = 0


class QueryResponse(BaseModel):
    """Full response bundle when all meshes are cached."""

    status: str = "complete"
    report: ReportResponse
    statcast: dict
    viewer: ViewerResponse
    query: dict


class ProgressResponse(BaseModel):
    """Returned when MLX inference is needed."""

    status: str = "processing"
    token: str
    pitches_needing_inference: list[str]
    total_pitches: int
    eta_seconds: int | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("", response_model=QueryResponse | ProgressResponse)
def submit_query(
    req: QueryRequest,
    job_store: dict = Depends(get_job_store),
):
    """Submit a natural language mechanics query.

    Returns the full response bundle if all meshes are cached.
    Returns a progress token if MLX inference is needed.
    """
    import os
    from backend.app.data.pitch_db import PitchDB
    from backend.app.query.parser import QueryParser
    from backend.app.query.orchestrator import QueryOrchestrator, QueryResult, ProgressToken
    from backend.app.reports.diagnostic import DiagnosticEngine, create_provider

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY not set. The query parser requires an Anthropic API key.",
        )

    db = PitchDB()
    parser = QueryParser(api_key=anthropic_key)

    # Prefer local Gemma4 if mlx-vlm is available, else fall back to Claude
    try:
        engine = DiagnosticEngine(provider=create_provider("gemma4"))
    except Exception:
        engine = DiagnosticEngine(
            provider=create_provider("claude", api_key=anthropic_key)
        )

    orch = QueryOrchestrator(db=db, parser=parser, diagnostic_engine=engine)

    try:
        result = orch.execute(req.text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

    if isinstance(result, ProgressToken):
        # Store token for polling
        job_store[result.token] = {
            "status": "processing",
            "pitches": result.pitches_needing_inference,
            "total_pitches": result.total_pitches,
            "query_text": req.text,
        }
        # Estimate: ~490ms/frame * 370 frames * n_pitches
        n = len(result.pitches_needing_inference)
        eta = int(490 * 370 * n / 1000)
        return ProgressResponse(
            token=result.token,
            pitches_needing_inference=result.pitches_needing_inference,
            total_pitches=result.total_pitches,
            eta_seconds=eta,
        )

    # Full result
    return QueryResponse(
        report=ReportResponse(
            narrative=result.report.narrative,
            recommendations=result.report.recommendations,
            risk_flags=result.report.risk_flags,
            confidence=result.report.confidence,
            pitches_analyzed=result.report.pitches_analyzed,
        ),
        statcast=result.statcast,
        viewer=ViewerResponse(
            glb_url=result.viewer["glb_url"],
            phase_markers=result.viewer.get("phase_markers", {}),
            total_frames=result.viewer.get("total_frames", 0),
        ),
        query=result.query,
    )


@router.get("/{token}/status")
def check_query_status(
    token: str,
    job_store: dict = Depends(get_job_store),
):
    """Poll for query completion status."""
    job = job_store.get(token)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown token: {token!r}")

    if job["status"] == "complete":
        result = job.get("result")
        # Clean up after delivery
        del job_store[token]
        return {"status": "complete", "result": result}

    if job["status"] == "error":
        error = job.get("error", "unknown error")
        del job_store[token]
        raise HTTPException(status_code=500, detail=error)

    return {
        "status": "processing",
        "token": token,
        "progress": job.get("progress"),
    }
