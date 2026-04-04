"""Generate synthetic demo data for SamPlaysBaseball.

Creates three pitchers with realistic-looking (but entirely synthetic) joint motion
data, baselines, and analysis results. Saves everything to demo/data/ in the same
JSON format that the real pipeline produces.

Run from repo root:
    python demo/generate_synthetic.py

No GPU required. Dependencies: numpy, pydantic (both in requirements.txt).
"""

from __future__ import annotations

import json
import math
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

DEMO_DATA_DIR = Path(__file__).parent / "data"

# ---------------------------------------------------------------------------
# Pitcher definitions
# ---------------------------------------------------------------------------

PITCHERS = [
    {
        "pitcher_id": "sam_torres",
        "name": "Sam Torres",
        "handedness": "R",
        "pitch_types": ["FF", "SL", "CH"],
        "pitch_counts": {"FF": 8, "SL": 5, "CH": 4},
    },
    {
        "pitcher_id": "jake_kim",
        "name": "Jake Kim",
        "handedness": "L",
        "pitch_types": ["FF", "CU", "CH"],
        "pitch_counts": {"FF": 7, "CU": 6, "CH": 4},
    },
    {
        "pitcher_id": "demo_pitcher_01",
        "name": "Demo Pitcher",
        "handedness": "R",
        "pitch_types": ["FF", "SL"],
        "pitch_counts": {"FF": 10, "SL": 6},
    },
]

# Velocity ranges by pitch type (mph)
VELOCITY_RANGES = {
    "FF": (90, 96),
    "SL": (82, 87),
    "CH": (83, 88),
    "CU": (73, 79),
    "SI": (89, 94),
    "FC": (86, 91),
}

# Spin rate ranges (rpm)
SPIN_RANGES = {
    "FF": (2100, 2400),
    "SL": (2300, 2600),
    "CH": (1700, 1900),
    "CU": (2400, 2800),
    "SI": (1900, 2200),
    "FC": (2200, 2500),
}

NUM_FRAMES = 60
NUM_JOINTS = 127
NUM_JOINTS_MHR70 = 70
POSE_DIM = 136
SHAPE_DIM = 45


# ---------------------------------------------------------------------------
# Synthetic motion generation
# ---------------------------------------------------------------------------

def _make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _generate_joints(rng: np.random.Generator, pitch_type: str) -> np.ndarray:
    """Generate plausible synthetic joint positions (T, 127, 3) in meters.

    Uses sinusoidal curves to simulate the kinematic chain of a pitching motion.
    Joint positions are not anatomically precise but follow realistic movement
    envelopes for a throwing motion.
    """
    T = NUM_FRAMES
    J = NUM_JOINTS
    t = np.linspace(0, 2 * math.pi, T)

    # Base pose: joints roughly distributed around origin
    base = rng.uniform(-0.5, 0.5, size=(J, 3)).astype(np.float32)

    # Apply a pitching-motion envelope: sinusoidal movement along primary axes
    motion_amp = rng.uniform(0.02, 0.15, size=(J, 3)).astype(np.float32)
    phase_offset = rng.uniform(0, 2 * math.pi, size=(J, 3)).astype(np.float32)

    # Different pitch types have slightly different motion signatures
    freq = 1.0 if pitch_type in ("FF", "SI") else 0.85

    # Broadcast t over (T, J, 3)
    t_broadcast = t[:, np.newaxis, np.newaxis]  # (T, 1, 1)
    motion = motion_amp[np.newaxis] * np.sin(freq * t_broadcast + phase_offset[np.newaxis])

    joints = base[np.newaxis] + motion  # (T, J, 3)
    return joints.astype(np.float32)


def _generate_joints_mhr70(full_joints: np.ndarray) -> np.ndarray:
    """Derive MHR70 skeleton from full joints by selecting a subset of indices."""
    # Select the first 70 joint indices as a simplified subset
    return full_joints[:, :NUM_JOINTS_MHR70, :].copy()


def _generate_pose_params(rng: np.random.Generator, T: int) -> np.ndarray:
    """Generate pose parameters (T, 136)."""
    base = rng.uniform(-0.3, 0.3, size=(POSE_DIM,)).astype(np.float32)
    noise = rng.normal(0, 0.02, size=(T, POSE_DIM)).astype(np.float32)
    return (base[np.newaxis] + noise).astype(np.float32)


def _generate_shape_params(rng: np.random.Generator) -> np.ndarray:
    """Generate shape parameters (45,) — constant per pitcher."""
    return rng.normal(0, 0.1, size=(SHAPE_DIM,)).astype(np.float32)


def _make_metadata(pitcher_id: str, pitch_id: str, pitch_type: str,
                   game_date: datetime, pitch_number: int,
                   handedness: str, rng: np.random.Generator) -> dict:
    vlo, vhi = VELOCITY_RANGES.get(pitch_type, (85, 92))
    slo, shi = SPIN_RANGES.get(pitch_type, (2000, 2400))
    # Plate location — roughly in the strike zone
    plate_x = float(rng.uniform(-0.7, 0.7))
    plate_z = float(rng.uniform(1.5, 3.5))
    results = ["strike", "ball", "foul", "in_play_out", "in_play_hit"]
    return {
        "pitch_id": pitch_id,
        "pitcher_id": pitcher_id,
        "game_date": game_date.isoformat(),
        "inning": int(rng.integers(1, 10)),
        "pitch_number": pitch_number,
        "pitch_type": pitch_type,
        "velocity_mph": float(rng.uniform(vlo, vhi)),
        "spin_rate_rpm": float(rng.uniform(slo, shi)),
        "plate_x": plate_x,
        "plate_z": plate_z,
        "result": rng.choice(results),
        "video_path": None,
        "frame_start": 0,
        "frame_end": NUM_FRAMES - 1,
    }


# ---------------------------------------------------------------------------
# Analysis generation
# ---------------------------------------------------------------------------

def _make_analysis(pitcher_id: str, pitch_id: str, pitch_type: str,
                   pitch_number: int, rng: np.random.Generator) -> list[dict]:
    """Generate synthetic analysis results for all 6 modules."""
    results = []

    # 1. Baseline comparison
    z = float(rng.normal(0.8, 0.4))
    results.append({
        "pitch_id": pitch_id,
        "pitcher_id": pitcher_id,
        "module": "baseline-comparison",
        "confidence": float(rng.uniform(0.75, 0.99)),
        "notes": None,
        "pitch_type": pitch_type,
        "overall_z_score": z,
        "top_deviations": [
            {"joint_index": int(i), "joint_name": f"joint_{i}", "deviation_mm": float(rng.uniform(2, 15)), "z_score": float(rng.uniform(0.5, 2.5))}
            for i in rng.choice(127, size=5, replace=False)
        ],
        "is_outlier": bool(abs(z) > 2.5),
        "outlier_threshold": 2.5,
    })

    # 2. Tipping detection
    sep = float(rng.uniform(0.1, 0.5))
    results.append({
        "pitch_id": pitch_id,
        "pitcher_id": pitcher_id,
        "module": "tipping-detection",
        "confidence": float(rng.uniform(0.75, 0.99)),
        "notes": None,
        "tip_signals": [],
        "max_separation_score": sep,
        "is_tipping": bool(sep > 0.7),
        "tipping_threshold": 0.7,
    })

    # 3. Fatigue tracking
    fatigue = max(0.0, min(1.0, float(pitch_number / 30 + rng.normal(0, 0.05))))
    results.append({
        "pitch_id": pitch_id,
        "pitcher_id": pitcher_id,
        "module": "fatigue-tracking",
        "confidence": float(rng.uniform(0.75, 0.99)),
        "notes": None,
        "pitch_number_in_game": pitch_number,
        "markers": [
            {
                "metric_name": "arm_slot_drop_deg",
                "value": float(rng.uniform(0, 3)),
                "baseline_value": 0.0,
                "pct_change": float(rng.uniform(-5, 5)),
            }
        ],
        "fatigue_score": fatigue,
        "is_fatigued": bool(fatigue > 0.6),
        "fatigue_threshold": 0.6,
    })

    # 4. Command analysis
    cmd = float(rng.uniform(0.4, 0.95))
    results.append({
        "pitch_id": pitch_id,
        "pitcher_id": pitcher_id,
        "module": "command-analysis",
        "confidence": float(rng.uniform(0.75, 0.99)),
        "notes": None,
        "pitch_type": pitch_type,
        "plate_x": float(rng.uniform(-0.7, 0.7)),
        "plate_z": float(rng.uniform(1.5, 3.5)),
        "release_x_deviation": float(rng.uniform(1, 12)),
        "release_z_deviation": float(rng.uniform(1, 12)),
        "command_score": cmd,
        "zone": int(rng.integers(1, 10)),
    })

    # 5. Arm slot drift
    slot = float(rng.uniform(25, 60))
    drift = float(rng.normal(0, 1.5))
    results.append({
        "pitch_id": pitch_id,
        "pitcher_id": pitcher_id,
        "module": "arm-slot-drift",
        "confidence": float(rng.uniform(0.75, 0.99)),
        "notes": None,
        "arm_slot_degrees": slot,
        "baseline_arm_slot_degrees": slot - drift,
        "drift_degrees": drift,
        "cumulative_drift_degrees": float(rng.uniform(0, 5)),
        "is_significant_drift": bool(abs(drift) > 3.0),
        "drift_threshold_degrees": 3.0,
    })

    # 6. Timing analysis
    timing = float(rng.uniform(0.65, 0.98))
    results.append({
        "pitch_id": pitch_id,
        "pitcher_id": pitcher_id,
        "module": "timing-analysis",
        "confidence": float(rng.uniform(0.75, 0.99)),
        "notes": None,
        "pitch_type": pitch_type,
        "events": [
            {"event_name": "foot_plant", "frame": 15, "time_ms": 250.0, "baseline_frame": 15, "frame_delta": 0},
            {"event_name": "max_hip_rotation", "frame": 35, "time_ms": 583.0, "baseline_frame": 34, "frame_delta": 1},
            {"event_name": "ball_release", "frame": 48, "time_ms": 800.0, "baseline_frame": 47, "frame_delta": 1},
        ],
        "timing_score": timing,
        "is_timing_issue": bool(timing < 0.7),
    })

    return results


# ---------------------------------------------------------------------------
# Baseline generation
# ---------------------------------------------------------------------------

def _make_baseline(pitcher_id: str, pitcher_name: str, handedness: str,
                   pitches_by_type: dict[str, list[np.ndarray]],
                   shape_params: np.ndarray) -> dict:
    """Build a PitcherBaseline dict from collected pitch joint arrays."""
    by_pitch_type = {}
    for pt, joint_list in pitches_by_type.items():
        # Stack all pitches for this type: (N, T, 127, 3)
        stacked = np.stack(joint_list, axis=0)  # (N, T, 127, 3)
        # Flatten time dimension: (N*T, 127, 3)
        flat = stacked.reshape(-1, NUM_JOINTS, 3)
        mean = flat.mean(axis=0)   # (127, 3)
        std = flat.std(axis=0) + 1e-6  # (127, 3)

        joint_stats = [
            {"mean": mean[j].tolist(), "std": std[j].tolist()}
            for j in range(NUM_JOINTS)
        ]

        # MHR70
        mean_mhr70 = mean[:NUM_JOINTS_MHR70]
        std_mhr70 = std[:NUM_JOINTS_MHR70]
        joint_stats_mhr70 = [
            {"mean": mean_mhr70[j].tolist(), "std": std_mhr70[j].tolist()}
            for j in range(NUM_JOINTS_MHR70)
        ]

        # Pose param stats (approximate with zeros for baseline)
        pose_mean = [0.0] * POSE_DIM
        pose_std = [0.05] * POSE_DIM

        by_pitch_type[pt] = {
            "pitch_type": pt,
            "sample_count": len(joint_list),
            "joint_stats": joint_stats,
            "joint_stats_mhr70": joint_stats_mhr70,
            "pose_param_stats": {"mean": pose_mean, "std": pose_std},
            "shape_params_mean": shape_params.tolist(),
        }

    return {
        "pitcher_id": pitcher_id,
        "pitcher_name": pitcher_name,
        "handedness": handedness,
        "by_pitch_type": by_pitch_type,
        "shape_params_mean": shape_params.tolist(),
        "shape_params_std": [0.05] * SHAPE_DIM,
    }


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_pitcher(pitcher_def: dict, seed: int) -> None:
    pitcher_id = pitcher_def["pitcher_id"]
    name = pitcher_def["name"]
    handedness = pitcher_def["handedness"]
    pitch_counts = pitcher_def["pitch_counts"]

    rng = _make_rng(seed)

    pitcher_dir = DEMO_DATA_DIR / pitcher_id
    analysis_dir = pitcher_dir / "analysis"
    pitcher_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Pitcher: {name} ({pitcher_id})")

    shape_params = _generate_shape_params(rng)
    pitches_by_type: dict[str, list[np.ndarray]] = {}
    game_date = datetime(2024, 6, 15, 19, 5, 0)
    pitch_number = 1

    for pt, count in pitch_counts.items():
        pitches_by_type[pt] = []
        for i in range(count):
            pitch_id = str(uuid.uuid4())
            joints = _generate_joints(rng, pt)
            joints_mhr70 = _generate_joints_mhr70(joints)
            pose_params = _generate_pose_params(rng, NUM_FRAMES)

            pitches_by_type[pt].append(joints)

            metadata = _make_metadata(
                pitcher_id, pitch_id, pt, game_date + timedelta(minutes=pitch_number * 3),
                pitch_number, handedness, rng
            )

            pitch_data = {
                "metadata": metadata,
                "joints": joints.tolist(),
                "joints_mhr70": joints_mhr70.tolist(),
                "pose_params": pose_params.tolist(),
                "shape_params": shape_params.tolist(),
                "num_frames": NUM_FRAMES,
                "skeleton_type": "both",
            }

            pitch_path = pitcher_dir / f"{pitch_id}.json"
            pitch_path.write_text(json.dumps(pitch_data))

            analysis_results = _make_analysis(pitcher_id, pitch_id, pt, pitch_number, rng)
            analysis_path = analysis_dir / f"{pitch_id}.json"
            analysis_path.write_text(json.dumps(analysis_results))

            pitch_number += 1
            print(f"    {pt} pitch {i+1}/{count}: {pitch_id[:8]}...")

    # Baseline
    baseline = _make_baseline(pitcher_id, name, handedness, pitches_by_type, shape_params)
    baseline_path = pitcher_dir / "baseline.json"
    baseline_path.write_text(json.dumps(baseline))
    print(f"    Baseline saved ({len(pitches_by_type)} pitch types)")


def main() -> None:
    print("SamPlaysBaseball — synthetic demo data generator")
    print(f"Output: {DEMO_DATA_DIR.resolve()}")
    print()

    DEMO_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for idx, pitcher_def in enumerate(PITCHERS):
        generate_pitcher(pitcher_def, seed=idx * 42 + 7)

    print()
    print("Done. Generated data for:")
    for p in PITCHERS:
        total = sum(p["pitch_counts"].values())
        print(f"  {p['name']} ({p['pitcher_id']}): {total} pitches — {', '.join(p['pitch_types'])}")
    print()
    print("Start the demo with:")
    print("  python demo/launcher.py")
    print("  — or —")
    print("  cd demo && docker compose up")


if __name__ == "__main__":
    main()
