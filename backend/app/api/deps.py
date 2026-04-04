"""Shared FastAPI dependencies to avoid circular imports."""

from __future__ import annotations

from fastapi import Request

from backend.app.models.storage import StorageLayer


def get_storage(request: Request) -> StorageLayer:
    """Return the shared StorageLayer from app state."""
    return request.app.state.storage


def get_job_store(request: Request) -> dict:
    """Return the in-memory job store from app state."""
    return request.app.state.job_store


def get_job_store_from_app(app) -> dict:
    """Return job store from a FastAPI app instance (for WebSocket)."""
    return app.state.job_store
