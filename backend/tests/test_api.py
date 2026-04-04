"""API endpoint tests using FastAPI TestClient."""

from __future__ import annotations

import io
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app.models.pitch import PitchData, PitchMetadata
from backend.app.models.baseline import PitcherBaseline, PitchTypeBaseline, JointStats, PoseParamStats
from backend.app.models.storage import StorageLayer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_pitch_data(pitch_id: str = "pitch-1", pitcher_id: str = "pitcher-1") -> PitchData:
    """Minimal valid PitchData for testing."""
    meta = PitchMetadata(
        pitch_id=pitch_id,
        pitcher_id=pitcher_id,
        game_date=datetime(2024, 6, 1),
        inning=1,
        pitch_number=1,
        pitch_type="FF",
        velocity_mph=94.0,
    )
    T = 3
    joints = [[[0.0, 0.0, 0.0]] * 127] * T
    joints_mhr70 = [[[0.0, 0.0, 0.0]] * 70] * T
    pose_params = [[0.0] * 136] * T
    shape_params = [0.0] * 45
    return PitchData(
        metadata=meta,
        joints=joints,
        joints_mhr70=joints_mhr70,
        pose_params=pose_params,
        shape_params=shape_params,
    )


def make_baseline(pitcher_id: str = "pitcher-1") -> PitcherBaseline:
    joint_stats = [JointStats(mean=[0.0, 0.0, 0.0], std=[0.1, 0.1, 0.1]) for _ in range(127)]
    pose_stats = PoseParamStats(mean=[0.0] * 136, std=[0.1] * 136)
    pt_baseline = PitchTypeBaseline(
        pitch_type="FF",
        sample_count=20,
        joint_stats=joint_stats,
        pose_param_stats=pose_stats,
        shape_params_mean=[0.0] * 45,
    )
    return PitcherBaseline(
        pitcher_id=pitcher_id,
        pitcher_name="Test Pitcher",
        handedness="R",
        by_pitch_type={"FF": pt_baseline},
    )


@pytest.fixture
def storage_with_data(tmp_path):
    """StorageLayer with one pitcher and one pitch loaded."""
    db = tmp_path / "test.db"
    parquet = tmp_path / "parquet"
    storage = StorageLayer(db_path=db, parquet_dir=parquet)

    baseline = make_baseline()
    storage.save_baseline(baseline)

    pitch = make_pitch_data()
    storage.save_pitch(pitch)

    return storage, "pitcher-1", "pitch-1"


@pytest.fixture
def test_client(storage_with_data):
    """TestClient with mocked storage."""
    storage, pitcher_id, pitch_id = storage_with_data
    app = create_app()

    # Override lifespan state before client starts
    with TestClient(app) as client:
        # Inject real storage via app state
        client.app.state.storage = storage
        client.app.state.job_store = {}
        yield client, pitcher_id, pitch_id


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_list_pitchers(test_client):
    client, pitcher_id, _ = test_client
    resp = client.get("/api/pitchers")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert any(p["pitcher_id"] == pitcher_id for p in data)


def test_get_pitcher_profile(test_client):
    client, pitcher_id, _ = test_client
    resp = client.get(f"/api/pitchers/{pitcher_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["pitcher_id"] == pitcher_id
    assert "pitch_types" in data


def test_get_pitcher_not_found(test_client):
    client, _, _ = test_client
    resp = client.get("/api/pitchers/nonexistent")
    assert resp.status_code == 404


def test_get_pitch(test_client):
    client, _, pitch_id = test_client
    resp = client.get(f"/api/pitches/{pitch_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["pitch_id"] == pitch_id
    assert "joints" in data


def test_get_pitch_not_found(test_client):
    client, _, _ = test_client
    resp = client.get("/api/pitches/nonexistent-pitch")
    assert resp.status_code == 404


def test_get_pitch_mesh(test_client):
    client, _, pitch_id = test_client
    resp = client.get(f"/api/pitches/{pitch_id}/mesh")
    assert resp.status_code == 200
    data = resp.json()
    assert data["pitch_id"] == pitch_id
    assert "frames" in data
    assert len(data["frames"]) == 3  # T=3


def test_upload_video(test_client):
    client, pitcher_id, _ = test_client
    video_bytes = b"fake video content"
    resp = client.post(
        "/api/upload",
        data={
            "pitcher_id": pitcher_id,
            "game_date": "2024-06-01",
            "inning": "1",
            "pitch_number": "5",
            "pitch_type": "SL",
        },
        files={"video": ("test.mp4", io.BytesIO(video_bytes), "video/mp4")},
    )
    assert resp.status_code == 202
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "queued"


def test_get_job_status(test_client):
    client, pitcher_id, _ = test_client
    # First upload to create a job
    resp = client.post(
        "/api/upload",
        data={
            "pitcher_id": pitcher_id,
            "game_date": "2024-06-01",
            "inning": "1",
            "pitch_number": "6",
            "pitch_type": "CH",
        },
        files={"video": ("test.mp4", io.BytesIO(b"fake"), "video/mp4")},
    )
    job_id = resp.json()["job_id"]
    resp2 = client.get(f"/api/jobs/{job_id}")
    assert resp2.status_code == 200
    data = resp2.json()
    assert data["job_id"] == job_id
    assert data["status"] in ("queued", "processing", "completed")


def test_get_job_not_found(test_client):
    client, _, _ = test_client
    resp = client.get("/api/jobs/nonexistent-job")
    assert resp.status_code == 404


def test_analysis_tipping(test_client):
    client, pitcher_id, _ = test_client
    resp = client.get(f"/api/analysis/tipping/{pitcher_id}")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_compare_pitches(test_client):
    """Compare a pitch with itself (same ID both sides)."""
    client, _, pitch_id = test_client
    resp = client.get(f"/api/compare?pitch_a={pitch_id}&pitch_b={pitch_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert "pitch_a" in data
    assert "pitch_b" in data


def test_compare_pitch_not_found(test_client):
    client, _, pitch_id = test_client
    resp = client.get(f"/api/compare?pitch_a={pitch_id}&pitch_b=nonexistent")
    assert resp.status_code == 404


def test_get_report(test_client):
    client, pitcher_id, _ = test_client
    resp = client.get(f"/api/reports/{pitcher_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["pitcher_id"] == pitcher_id
    assert "sections" in data
    assert "narrative" in data


def test_get_baseline_summary(test_client):
    client, pitcher_id, _ = test_client
    resp = client.get(f"/api/analysis/baseline/{pitcher_id}/FF")
    assert resp.status_code == 200
    data = resp.json()
    assert data["pitcher_id"] == pitcher_id
    assert data["pitch_type"] == "FF"
    assert data["sample_count"] == 20
