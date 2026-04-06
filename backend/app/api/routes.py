"""Central router registration."""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.api import analysis, compare, demo, pitchers, pitches, query, reports, upload, websocket

api_router = APIRouter()

api_router.include_router(pitchers.router)
api_router.include_router(pitches.router)
api_router.include_router(analysis.router)
api_router.include_router(upload.router)
api_router.include_router(compare.router)
api_router.include_router(demo.router)
api_router.include_router(websocket.router)
api_router.include_router(reports.router)
api_router.include_router(query.router)
