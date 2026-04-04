"""Process historical legend footage through the SamPlaysBaseball pipeline.

Run this script on a GPU machine. Lower-quality broadcast footage from the
1990s and 2000s is handled via the --quality-flag option, which enables
additional temporal smoothing and relaxed confidence thresholds.

Usage:
    python demo/legends/process_legends.py \\
        --input-dir /path/to/rivera_footage \\
        --pitcher-id rivera \\
        --quality-flag broadcast \\
        --device cuda

Supported pitcher IDs: rivera, pedro_martinez, kershaw, maddux, randy_johnson

Requirements (GPU machine):
    pip install -r requirements.txt
    # plus: torch>=2.0, torchvision, sam3d (internal package)

Output: JSON files in demo/data/legends/{pitcher_id}/ in standard PitchData format.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

LEGEND_PITCHER_IDS = {
    "rivera": "Mariano Rivera",
    "pedro_martinez": "Pedro Martinez",
    "kershaw": "Clayton Kershaw",
    "maddux": "Greg Maddux",
    "randy_johnson": "Randy Johnson",
}

LEGEND_HANDEDNESS = {
    "rivera": "R",
    "pedro_martinez": "R",
    "kershaw": "L",
    "maddux": "R",
    "randy_johnson": "L",
}

# Typical pitch types for each legend
LEGEND_PITCH_TYPES = {
    "rivera": ["FC"],            # Nearly exclusively cutter
    "pedro_martinez": ["FF", "CH", "CU"],
    "kershaw": ["FF", "CU", "SL"],
    "maddux": ["FF", "CH", "SL", "CU"],
    "randy_johnson": ["FF", "SL"],
}

# Smoothing window sizes by quality level
SMOOTHING_WINDOWS = {
    "hd": 3,          # Modern HD footage
    "broadcast": 7,   # Standard broadcast (30fps, compressed)
    "archival": 11,   # Low-quality archive footage
}

# SAM 3D Body confidence thresholds by quality level
CONFIDENCE_THRESHOLDS = {
    "hd": 0.7,
    "broadcast": 0.5,
    "archival": 0.35,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Process legend footage through the pipeline")
    p.add_argument(
        "--input-dir", required=True, type=Path,
        help="Directory of MP4 files for this pitcher"
    )
    p.add_argument(
        "--pitcher-id", required=True, choices=list(LEGEND_PITCHER_IDS.keys()),
        help="Legend pitcher ID"
    )
    p.add_argument(
        "--quality-flag", choices=["hd", "broadcast", "archival"], default="broadcast",
        help="Video quality level — controls smoothing and confidence thresholds"
    )
    p.add_argument(
        "--pitch-type", default=None,
        help="Pitch type label (default: first in legend's known repertoire)"
    )
    p.add_argument("--device", default="cuda", help="Torch device (cuda, mps, cpu)")
    p.add_argument("--fps", type=int, default=30, help="Source video frame rate")
    p.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).parent.parent / "data" / "legends",
        help="Output directory"
    )
    return p.parse_args()


def check_gpu_deps() -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        print("ERROR: torch not found. Run on a GPU machine with requirements.txt installed.")
        sys.exit(1)

    try:
        import sam3d  # noqa: F401
    except ImportError:
        print("ERROR: sam3d not found. Install the SAM 3D Body package.")
        sys.exit(1)


def load_pipeline(device: str, quality_flag: str) -> tuple:
    """Load pipeline with quality-appropriate settings."""
    from backend.app.pipeline.video import VideoPipeline
    from backend.app.pipeline.inference import InferencePipeline

    smoothing_window = SMOOTHING_WINDOWS[quality_flag]
    confidence_threshold = CONFIDENCE_THRESHOLDS[quality_flag]

    print(f"  Quality: {quality_flag} (smoothing window={smoothing_window}, "
          f"confidence threshold={confidence_threshold})")

    video_pipeline = VideoPipeline(fps_target=30)
    sam_pipeline = InferencePipeline(
        device=device,
        confidence_threshold=confidence_threshold,
    )
    return video_pipeline, sam_pipeline, smoothing_window


def process_legend_video(
    video_path: Path,
    pitcher_id: str,
    pitch_type: str,
    pitch_number: int,
    video_pipeline,
    sam_pipeline,
    smoothing_window: int,
) -> dict:
    """Process one legend video. Returns PitchData-compatible dict."""
    from backend.app.pipeline.smoothing import smooth_joints
    from backend.app.pipeline.alignment import align_to_mound
    from backend.app.models.pitch import PitchMetadata, PitchData

    pitch_id = str(uuid.uuid4())
    print(f"  [{pitch_number}] {video_path.name} → {pitch_id[:8]}...")

    # Decode video
    frames = video_pipeline.decode(video_path)

    # SAM inference with relaxed confidence threshold for broadcast footage
    raw_joints, pose_params, shape_params = sam_pipeline.run(frames)

    # Apply heavier temporal smoothing for broadcast footage
    joints = smooth_joints(raw_joints, window=smoothing_window)
    joints_mhr70 = joints[:, :70, :]

    # Mound alignment
    joints = align_to_mound(joints)

    # Estimate era from pitcher ID for metadata
    era_dates = {
        "rivera": datetime(1998, 10, 1),
        "pedro_martinez": datetime(1999, 9, 15),
        "kershaw": datetime(2014, 6, 20),
        "maddux": datetime(1995, 8, 10),
        "randy_johnson": datetime(2002, 7, 4),
    }
    game_date = era_dates.get(pitcher_id, datetime(2000, 1, 1))

    metadata = PitchMetadata(
        pitch_id=pitch_id,
        pitcher_id=pitcher_id,
        game_date=game_date,
        inning=1,
        pitch_number=pitch_number,
        pitch_type=pitch_type,
        video_path=str(video_path),
        frame_start=0,
        frame_end=len(joints) - 1,
    )

    return PitchData.from_numpy(
        metadata=metadata,
        joints=joints,
        pose_params=pose_params,
        shape_params=shape_params,
        joints_mhr70=joints_mhr70,
    ).model_dump()


def main() -> None:
    args = parse_args()
    check_gpu_deps()

    pitcher_name = LEGEND_PITCHER_IDS[args.pitcher_id]
    handedness = LEGEND_HANDEDNESS[args.pitcher_id]
    pitch_type = args.pitch_type or LEGEND_PITCH_TYPES[args.pitcher_id][0]

    video_files = sorted(args.input_dir.glob("*.mp4"))
    if not video_files:
        print(f"No MP4 files found in {args.input_dir}")
        sys.exit(1)

    print(f"SamPlaysBaseball — legend processor")
    print(f"  Legend:   {pitcher_name} ({args.pitcher_id}, {handedness}H)")
    print(f"  Videos:   {len(video_files)} MP4s")
    print(f"  Output:   {args.output_dir}")
    print()

    video_pipeline, sam_pipeline, smoothing_window = load_pipeline(
        args.device, args.quality_flag
    )

    pitcher_dir = args.output_dir / args.pitcher_id
    pitcher_dir.mkdir(parents=True, exist_ok=True)

    for idx, video_path in enumerate(video_files):
        pitch_data = process_legend_video(
            video_path=video_path,
            pitcher_id=args.pitcher_id,
            pitch_type=pitch_type,
            pitch_number=idx + 1,
            video_pipeline=video_pipeline,
            sam_pipeline=sam_pipeline,
            smoothing_window=smoothing_window,
        )
        pitch_id = pitch_data["metadata"]["pitch_id"]
        (pitcher_dir / f"{pitch_id}.json").write_text(json.dumps(pitch_data))

    print()
    print(f"Processed {len(video_files)} pitches for {pitcher_name}.")
    print(f"NOTE: Accuracy is reduced on {args.quality_flag}-quality footage.")
    print(f"See demo/legends/talking_points.json for accuracy expectations.")


if __name__ == "__main__":
    main()
