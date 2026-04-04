"""Shared helpers for analysis module tests."""

from __future__ import annotations

import numpy as np

from backend.app.pipeline.features import BiomechFeatures
from backend.app.pipeline.phases import PitchPhases
from backend.app.pipeline.kinetics import KineticChainTiming


def make_features(
    *,
    max_shoulder_er_deg: float = 170.0,
    hip_shoulder_sep_deg: float = 45.0,
    stride_length_normalized: float = 0.85,
    release_point: list[float] | None = None,
    foot_plant: int = 5,
    mer: int = 20,
    release: int = 30,
    total_frames: int = 40,
    pelvis_peak: int = 6,
    trunk_peak: int = 8,
    shoulder_peak: int = 10,
    elbow_peak: int = 12,
    wrist_peak: int = 14,
    rng: np.random.Generator | None = None,
) -> BiomechFeatures:
    if rng is None:
        rng = np.random.default_rng(42)
    T = total_frames
    rp = np.array(release_point if release_point is not None else [0.5, 1.8, 0.0])

    return BiomechFeatures(
        elbow_flexion=rng.uniform(60, 100, T),
        shoulder_abduction=rng.uniform(80, 100, T),
        shoulder_er=rng.uniform(140, 170, T),
        hip_flexion=rng.uniform(20, 40, T),
        knee_flexion=rng.uniform(10, 30, T),
        trunk_tilt=rng.uniform(-5, 5, T),
        trunk_rotation=rng.uniform(-10, 10, T),
        elbow_flexion_vel=rng.uniform(-50, 50, T),
        shoulder_er_vel=rng.uniform(-100, 100, T),
        trunk_rotation_vel=rng.uniform(-200, 200, T),
        max_shoulder_er_deg=max_shoulder_er_deg,
        hip_shoulder_sep_deg=hip_shoulder_sep_deg,
        stride_length_normalized=stride_length_normalized,
        release_point=rp,
        phases=PitchPhases(
            foot_plant=foot_plant,
            mer=mer,
            release=release,
            total_frames=total_frames,
        ),
        kinetic_chain=KineticChainTiming(
            pelvis_peak_frame=pelvis_peak,
            trunk_peak_frame=trunk_peak,
            shoulder_peak_frame=shoulder_peak,
            elbow_peak_frame=elbow_peak,
            wrist_peak_frame=wrist_peak,
        ),
    )
