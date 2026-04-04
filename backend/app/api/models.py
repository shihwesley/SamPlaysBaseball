"""API response models for the baseball analyzer endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class PitcherSummary(BaseModel):
    pitcher_id: str
    name: str | None = None
    handedness: str | None = None
    pitch_count: int = 0
    pitch_types: list[str] = Field(default_factory=list)


class PitcherProfile(BaseModel):
    pitcher_id: str
    name: str | None = None
    handedness: str | None = None
    pitch_types: list[str] = Field(default_factory=list)
    sample_counts: dict[str, int] = Field(default_factory=dict)


class PitchListItem(BaseModel):
    pitch_id: str
    pitcher_id: str
    game_date: str
    pitch_type: str | None = None
    velocity_mph: float | None = None
    result: str | None = None


class PitchResponse(BaseModel):
    pitch_id: str
    pitcher_id: str
    game_date: str
    pitch_type: str | None = None
    velocity_mph: float | None = None
    spin_rate_rpm: float | None = None
    plate_x: float | None = None
    plate_z: float | None = None
    result: str | None = None
    num_frames: int = 0
    joints: list[list[list[float]]] = Field(default_factory=list)
    joints_mhr70: list[list[list[float]]] | None = None
    pose_params: list[list[float]] = Field(default_factory=list)
    shape_params: list[float] = Field(default_factory=list)
    analysis_results: list[dict] = Field(default_factory=list)


class MeshJoint(BaseModel):
    x: float
    y: float
    z: float


class MeshFrame(BaseModel):
    joints: list[MeshJoint]


class MeshResponse(BaseModel):
    pitch_id: str
    frames: list[MeshFrame]


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "processing", "completed", "failed"] = "queued"
    progress: int = Field(default=0, ge=0, le=100)
    message: str | None = None
    pitch_id: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class UploadResponse(BaseModel):
    job_id: str
    status: str


class CompareResponse(BaseModel):
    pitch_a: PitchResponse
    pitch_b: PitchResponse


class BaselineSummary(BaseModel):
    pitcher_id: str
    pitch_type: str
    sample_count: int
