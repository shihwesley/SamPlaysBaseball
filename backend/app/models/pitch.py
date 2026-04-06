"""PitchData and PitchMetadata models.

SAM 3D Body outputs:
- joints: (T, 127, 3) — full model joints
- joints_mhr70: (T, 70, 3) — reduced MHR70 skeleton
- pose_params: (T, 136)
- shape_params: (45,)
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, field_serializer, model_validator


class PitchMetadata(BaseModel):
    pitch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    pitcher_id: str
    game_date: datetime
    inning: int
    pitch_number: int  # within game
    pitch_type: str  # FF, SL, CH, CU, SI, FC, etc.
    velocity_mph: float | None = None
    spin_rate_rpm: float | None = None
    # Statcast location
    plate_x: float | None = None
    plate_z: float | None = None
    result: str | None = None  # ball, strike, hit, etc.
    player_height_m: float | None = None  # roster height for scale correction
    video_path: str | None = None
    frame_start: int | None = None
    frame_end: int | None = None

    model_config = {}

    @field_serializer("game_date")
    def serialize_game_date(self, v: datetime) -> str:
        return v.isoformat()


class PitchData(BaseModel):
    """Full SAM 3D Body output for one pitch sequence."""

    metadata: PitchMetadata

    # Core SAM 3D Body outputs stored as nested lists for JSON serialization.
    # Shape: (T, 127, 3)
    joints: list[list[list[float]]]
    # Shape: (T, 70, 3) — MHR70 reduced skeleton
    joints_mhr70: list[list[list[float]]] | None = None
    # Shape: (T, 136)
    pose_params: list[list[float]]
    # Shape: (45,)
    shape_params: list[float]

    # Derived
    num_frames: int = Field(init=False, default=0)
    skeleton_type: Literal["full", "mhr70", "both"] = "both"

    @model_validator(mode="after")
    def set_num_frames(self) -> "PitchData":
        self.num_frames = len(self.joints)
        return self

    # ------------------------------------------------------------------
    # Numpy conversion helpers
    # ------------------------------------------------------------------

    def joints_array(self) -> np.ndarray:
        """Return joints as (T, 127, 3) float32 array."""
        return np.array(self.joints, dtype=np.float32)

    def joints_mhr70_array(self) -> np.ndarray | None:
        if self.joints_mhr70 is None:
            return None
        return np.array(self.joints_mhr70, dtype=np.float32)

    def pose_params_array(self) -> np.ndarray:
        """Return pose_params as (T, 136) float32 array."""
        return np.array(self.pose_params, dtype=np.float32)

    def shape_params_array(self) -> np.ndarray:
        """Return shape_params as (45,) float32 array."""
        return np.array(self.shape_params, dtype=np.float32)

    @classmethod
    def from_numpy(
        cls,
        metadata: PitchMetadata,
        joints: np.ndarray,
        pose_params: np.ndarray,
        shape_params: np.ndarray,
        joints_mhr70: np.ndarray | None = None,
    ) -> "PitchData":
        """Construct from raw numpy arrays."""
        skeleton_type: Literal["full", "mhr70", "both"] = (
            "both" if joints_mhr70 is not None else "full"
        )
        return cls(
            metadata=metadata,
            joints=joints.tolist(),
            joints_mhr70=joints_mhr70.tolist() if joints_mhr70 is not None else None,
            pose_params=pose_params.tolist(),
            shape_params=shape_params.tolist(),
            skeleton_type=skeleton_type,
        )

    def scale_to_height(self) -> "PitchData":
        """Apply height-based scale correction using roster data.

        Computes a uniform scale factor from the mesh's Y-span vs the
        player's known height, then rescales all joint positions.
        Returns self (mutates in place) for chaining.
        """
        height_m = self.metadata.player_height_m
        if height_m is None or self.num_frames == 0:
            return self

        joints = self.joints_array()  # (T, 127, 3)
        # Mesh height = max Y - min Y across all frames
        mesh_height = float(joints[:, :, 1].max() - joints[:, :, 1].min())
        if mesh_height < 0.1:
            return self

        scale = height_m / mesh_height
        self.joints = (joints * scale).tolist()

        if self.joints_mhr70 is not None:
            mhr70 = self.joints_mhr70_array()
            self.joints_mhr70 = (mhr70 * scale).tolist()

        return self
