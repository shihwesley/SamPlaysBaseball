"""Compare two pitch deliveries — biomechanical diff + optional 3D overlay.

Usage:
    from backend.app.analysis.compare_deliveries import compare_deliveries

    result = compare_deliveries(
        joints_a=kp_array_a,  # (T, 70, 3) from MLX or PyTorch
        joints_b=kp_array_b,
        fps=60.0,
        handedness="right",
        pitcher_height_m=1.93,
        labels=("Fastball #1", "Fastball #2"),
    )

    # result.summary  -> plain text comparison
    # result.diffs    -> per-feature numeric diffs
    # result.features_a / features_b -> full BiomechFeatures
    # result.aligned_a / aligned_b   -> phase-normalized (101, 70, 3) arrays

    # Optional: export comparison GLB
    result.export_glb("output/comparison.glb", mesh_frames_a, mesh_frames_b)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from backend.app.pipeline.features import BiomechFeatures, FeatureExtractor
from backend.app.pipeline.alignment import normalize_time_axis


@dataclass
class FeatureDiff:
    """Difference in one biomechanical feature between two pitches."""

    name: str
    value_a: float
    value_b: float
    diff: float        # b - a
    abs_diff: float
    unit: str = "deg"


@dataclass
class DeliveryComparison:
    """Result of comparing two pitch deliveries."""

    label_a: str
    label_b: str
    features_a: BiomechFeatures
    features_b: BiomechFeatures
    diffs: list[FeatureDiff]

    # Phase-normalized joint arrays for visualization (101, 70, 3)
    aligned_a: NDArray[np.float64] | None = None
    aligned_b: NDArray[np.float64] | None = None

    # Per-frame joint distance (101,) — mean joint displacement per normalized frame
    frame_distances: NDArray[np.float64] | None = None

    @property
    def summary(self) -> str:
        """Plain-text comparison summary."""
        lines = [f"Comparing: {self.label_a} vs {self.label_b}", ""]

        # Phase timing
        pa, pb = self.features_a.phases, self.features_b.phases
        lines.append("Phase timing:")
        lines.append(f"  Foot plant → MER:     {pa.mer - pa.foot_plant:3d} vs {pb.mer - pb.foot_plant:3d} frames")
        lines.append(f"  MER → Release:        {pa.release - pa.mer:3d} vs {pb.release - pb.mer:3d} frames")
        lines.append("")

        # Sorted by absolute difference
        lines.append("Feature differences (sorted by magnitude):")
        for d in sorted(self.diffs, key=lambda x: x.abs_diff, reverse=True):
            arrow = "↑" if d.diff > 0 else "↓" if d.diff < 0 else "="
            lines.append(
                f"  {d.name:28s}: {d.value_a:8.1f} vs {d.value_b:8.1f}  "
                f"({arrow} {d.abs_diff:.1f} {d.unit})"
            )

        # Kinetic chain
        lines.append("")
        kca = self.features_a.kinetic_chain
        kcb = self.features_b.kinetic_chain
        lines.append("Kinetic chain order:")
        for label, feat in [(self.label_a, kca), (self.label_b, kcb)]:
            chain = [
                (feat.pelvis_peak_frame, "pelvis"),
                (feat.trunk_peak_frame, "trunk"),
                (feat.shoulder_peak_frame, "shoulder"),
                (feat.elbow_peak_frame, "elbow"),
                (feat.wrist_peak_frame, "wrist"),
            ]
            order = " → ".join(name for _, name in sorted(chain))
            lines.append(f"  {label}: {order}")

        # Most divergent frame
        if self.frame_distances is not None:
            peak_frame = int(np.argmax(self.frame_distances))
            peak_dist = float(self.frame_distances[peak_frame])
            pct = peak_frame / len(self.frame_distances) * 100
            phase = "wind-up" if pct < 33 else "arm cocking" if pct < 66 else "acceleration"
            lines.append("")
            lines.append(
                f"Most divergent moment: {pct:.0f}% through delivery ({phase}), "
                f"{peak_dist:.3f}m avg joint displacement"
            )

        return "\n".join(lines)

    def export_glb(
        self,
        output_path: str,
        mesh_frames_a: list[NDArray] | None = None,
        mesh_frames_b: list[NDArray] | None = None,
    ) -> str | None:
        """Export a comparison GLB if mesh frames are provided."""
        if mesh_frames_a is None or mesh_frames_b is None:
            return None
        from backend.app.export.comparison import ComparisonGLBBuilder

        builder = ComparisonGLBBuilder()
        return builder.build(
            pitch1_frames=mesh_frames_a,
            pitch2_frames=mesh_frames_b,
            metadata1={"label": self.label_a},
            metadata2={"label": self.label_b},
            output_path=output_path,
        )

    def top_differences(self, n: int = 3) -> list[FeatureDiff]:
        """Return the N largest differences."""
        return sorted(self.diffs, key=lambda x: x.abs_diff, reverse=True)[:n]


def compare_deliveries(
    joints_a: NDArray[np.float64],
    joints_b: NDArray[np.float64],
    fps: float = 30.0,
    handedness: str = "right",
    pitcher_height_m: float | None = None,
    labels: tuple[str, str] = ("Pitch A", "Pitch B"),
) -> DeliveryComparison:
    """Compare two pitch deliveries end-to-end.

    Args:
        joints_a: (T1, 70, 3) MHR70 keypoints for first pitch
        joints_b: (T2, 70, 3) MHR70 keypoints for second pitch
        fps: video frame rate
        handedness: "right" or "left"
        pitcher_height_m: optional pitcher height for stride normalization
        labels: display names for the two pitches

    Returns:
        DeliveryComparison with features, diffs, aligned arrays, and summary.
    """
    joints_a = np.asarray(joints_a, dtype=np.float64)
    joints_b = np.asarray(joints_b, dtype=np.float64)

    extractor = FeatureExtractor(
        fps=fps, handedness=handedness, pitcher_height_m=pitcher_height_m,
    )
    fa = extractor.extract_from_array(joints_a)
    fb = extractor.extract_from_array(joints_b)

    # Build feature diffs
    comparisons = [
        ("Max shoulder ER", fa.max_shoulder_er_deg, fb.max_shoulder_er_deg, "deg"),
        ("Hip-shoulder sep", fa.hip_shoulder_sep_deg, fb.hip_shoulder_sep_deg, "deg"),
        ("Stride length (norm)", fa.stride_length_normalized, fb.stride_length_normalized, ""),
        ("Elbow flex range", _range(fa.elbow_flexion), _range(fb.elbow_flexion), "deg"),
        ("Shoulder ER range", _range(fa.shoulder_er), _range(fb.shoulder_er), "deg"),
        ("Trunk rotation range", _range(fa.trunk_rotation), _range(fb.trunk_rotation), "deg"),
        ("Peak elbow flex vel", _peak(fa.elbow_flexion_vel), _peak(fb.elbow_flexion_vel), "deg/s"),
        ("Peak shoulder ER vel", _peak(fa.shoulder_er_vel), _peak(fb.shoulder_er_vel), "deg/s"),
        ("Release X", float(fa.release_point[0]), float(fb.release_point[0]), "m"),
        ("Release Y", float(fa.release_point[1]), float(fb.release_point[1]), "m"),
        ("Release Z", float(fa.release_point[2]), float(fb.release_point[2]), "m"),
    ]

    diffs = [
        FeatureDiff(
            name=name, value_a=va, value_b=vb,
            diff=vb - va, abs_diff=abs(vb - va), unit=unit,
        )
        for name, va, vb, unit in comparisons
    ]

    # Phase-normalized alignment for spatial comparison
    aligned_a = normalize_time_axis(joints_a, fa.phases, n_samples=101)
    aligned_b = normalize_time_axis(joints_b, fb.phases, n_samples=101)

    # Per-frame mean joint displacement
    frame_dists = np.mean(
        np.linalg.norm(aligned_a - aligned_b, axis=2), axis=1
    )

    return DeliveryComparison(
        label_a=labels[0],
        label_b=labels[1],
        features_a=fa,
        features_b=fb,
        diffs=diffs,
        aligned_a=aligned_a,
        aligned_b=aligned_b,
        frame_distances=frame_dists,
    )


def compare_from_db(
    play_id_a: str,
    play_id_b: str,
    db_path: str = "data/pitches.db",
    mesh_dir: str = "data/meshes",
    fps: float = 60.0,
    handedness: str = "right",
    pitcher_height_m: float | None = None,
) -> DeliveryComparison:
    """Compare two pitches by play_id, loading joints from the database.

    Automatically uses pitch type + inning as labels.
    """
    from backend.app.data.pitch_db import PitchDB

    db = PitchDB(db_path, mesh_dir)
    try:
        mesh_a = db.load_mesh(play_id_a)
        mesh_b = db.load_mesh(play_id_b)
        if mesh_a is None or mesh_b is None:
            missing = play_id_a if mesh_a is None else play_id_b
            raise ValueError(f"No mesh data for pitch {missing}")

        pitch_a = db.get_pitch(play_id_a)
        pitch_b = db.get_pitch(play_id_b)

        label_a = f"{pitch_a['pitch_type'] or '??'} Inn {pitch_a['inning']}"
        label_b = f"{pitch_b['pitch_type'] or '??'} Inn {pitch_b['inning']}"

        return compare_deliveries(
            joints_a=mesh_a.joints_mhr70,
            joints_b=mesh_b.joints_mhr70,
            fps=fps,
            handedness=handedness,
            pitcher_height_m=pitcher_height_m,
            labels=(label_a, label_b),
        )
    finally:
        db.close()


def compare_pitch_types(
    type_a: str,
    type_b: str,
    pitcher_id: int | None = None,
    db_path: str = "data/pitches.db",
    mesh_dir: str = "data/meshes",
    fps: float = 60.0,
    handedness: str = "right",
    pitcher_height_m: float | None = None,
    sample_idx: tuple[int, int] = (0, 0),
) -> DeliveryComparison:
    """Compare two pitch types (e.g. 'FF' vs 'FC', 'CH' vs 'CU').

    Picks one sample of each type from the database. Use sample_idx
    to select which pitch of each type to use (default: first available).

    Args:
        type_a: Statcast pitch type code (FF, SI, FC, SL, CU, CH, etc.)
        type_b: Statcast pitch type code
        pitcher_id: optional — filter to one pitcher
        sample_idx: (idx_a, idx_b) — which pitch to pick from each type's list
    """
    from backend.app.data.pitch_db import PitchDB

    db = PitchDB(db_path, mesh_dir)
    try:
        pitches_a = db.get_by_type(type_a, pitcher_id=pitcher_id)
        pitches_b = db.get_by_type(type_b, pitcher_id=pitcher_id)

        # Filter to pitches that have mesh data
        pitches_a = [p for p in pitches_a if p.get("mesh_path")]
        pitches_b = [p for p in pitches_b if p.get("mesh_path")]

        if not pitches_a:
            raise ValueError(f"No pitches with mesh data for type '{type_a}'")
        if not pitches_b:
            raise ValueError(f"No pitches with mesh data for type '{type_b}'")

        idx_a = min(sample_idx[0], len(pitches_a) - 1)
        idx_b = min(sample_idx[1], len(pitches_b) - 1)

        pa = pitches_a[idx_a]
        pb = pitches_b[idx_b]

        mesh_a = db.load_mesh(pa["play_id"])
        mesh_b = db.load_mesh(pb["play_id"])

        label_a = f"{type_a} Inn {pa['inning']} #{idx_a + 1}"
        label_b = f"{type_b} Inn {pb['inning']} #{idx_b + 1}"

        return compare_deliveries(
            joints_a=mesh_a.joints_mhr70,
            joints_b=mesh_b.joints_mhr70,
            fps=fps,
            handedness=handedness,
            pitcher_height_m=pitcher_height_m,
            labels=(label_a, label_b),
        )
    finally:
        db.close()


def _range(arr: NDArray) -> float:
    return float(np.max(arr) - np.min(arr))


def _peak(arr: NDArray) -> float:
    return float(np.max(np.abs(arr)))
