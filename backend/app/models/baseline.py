"""PitcherBaseline models — per-pitcher, per-pitch-type statistical baselines."""

from __future__ import annotations

from pydantic import BaseModel, Field


class JointStats(BaseModel):
    """Mean and std for a single joint coordinate across a population of pitches."""

    mean: list[float]  # (3,) — x, y, z
    std: list[float]  # (3,)


class PoseParamStats(BaseModel):
    """Mean and std for pose parameters across frames."""

    mean: list[float]  # (136,)
    std: list[float]  # (136,)


class PitchTypeBaseline(BaseModel):
    """Statistical baseline for one pitcher throwing one pitch type."""

    pitch_type: str
    sample_count: int

    # Per-joint stats: list of 127 JointStats, one per joint
    joint_stats: list[JointStats]  # len 127

    # MHR70 reduced joint stats (optional)
    joint_stats_mhr70: list[JointStats] | None = None  # len 70

    # Pose param stats
    pose_param_stats: PoseParamStats

    # Shape params are per-pitcher, averaged over all pitch types
    shape_params_mean: list[float]  # (45,)


class PitcherBaseline(BaseModel):
    """All baselines for one pitcher."""

    pitcher_id: str
    pitcher_name: str | None = None
    handedness: str | None = None  # R or L
    # Map from pitch_type -> baseline stats
    by_pitch_type: dict[str, PitchTypeBaseline] = Field(default_factory=dict)

    # Shape params averaged over all pitches for this pitcher
    shape_params_mean: list[float] | None = None  # (45,)
    shape_params_std: list[float] | None = None  # (45,)

    def get_baseline(self, pitch_type: str) -> PitchTypeBaseline | None:
        return self.by_pitch_type.get(pitch_type)
