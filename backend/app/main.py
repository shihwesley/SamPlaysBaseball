"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.models.storage import StorageLayer

_DB_PATH = Path("./data/baseball.db")
_PARQUET_DIR = Path("./data/parquet")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialise storage and job store
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    app.state.storage = StorageLayer(db_path=_DB_PATH, parquet_dir=_PARQUET_DIR)
    app.state.job_store: dict = {}
    yield
    # Shutdown: nothing to do


def create_app() -> FastAPI:
    app = FastAPI(
        title="SamPlaysBaseball API",
        description="Baseball pitching mechanics analysis backend",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:3001"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from backend.app.api.routes import api_router

    app.include_router(api_router)

    return app


app = create_app()
