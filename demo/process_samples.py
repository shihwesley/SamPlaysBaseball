"""Process sample video files through the full SamPlaysBaseball pipeline.

Run this script on a GPU machine that has SAM 3D Body installed and the full
backend dependencies available (torch, sam3d, etc.).

Usage:
    python demo/process_samples.py \\
        --input-dir /path/to/sample_videos \\
        --output-dir demo/data \\
        --pitcher-id sam_torres \\
        --pitcher-name "Sam Torres" \\
        --handedness R

Input: directory of MP4 files, one file per pitch.
Output: JSON files in the same format as generate_synthetic.py, readable by the demo API.

Requirements (GPU machine):
    pip install -r requirements.txt
    # plus: torch>=2.0, torchvision, sam3d (internal package)
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Process sample MP4s through the full pipeline")
    p.add_argument("--input-dir", required=True, type=Path, help="Directory of MP4 pitch videos")
    p.add_argument("--output-dir", required=True, type=Path, help="Output directory for JSON data")
    p.add_argument("--pitcher-id", required=True, help="Pitcher ID (slug, e.g. sam_torres)")
    p.add_argument("--pitcher-name", required=True, help="Display name")
    p.add_argument("--handedness", choices=["R", "L"], default="R")
    p.add_argument("--pitch-type", default="FF", help="Pitch type label for all videos in this batch")
    p.add_argument("--device", default="cuda", help="Torch device (cuda, mps, cpu)")
    p.add_argument("--fps", type=int, default=30, help="Source video frame rate")
    return p.parse_args()


def check_gpu_deps() -> None:
    """Fail fast if GPU dependencies aren't available."""
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


def load_pipeline(device: str) -> tuple:
    """Load the inference pipeline. Returns (video_pipeline, sam_model)."""
    from backend.app.pipeline.video import VideoPipeline
    from backend.app.pipeline.inference import InferencePipeline

    print(f"  Loading SAM 3D Body model on {device}...")
    video_pipeline = VideoPipeline(fps_target=30)
    sam_pipeline = InferencePipeline(device=device)
    return video_pipeline, sam_pipeline


def process_video(
    video_path: Path,
    pitcher_id: str,
    pitch_type: str,
    pitch_number: int,
    video_pipeline,
    sam_pipeline,
    fps: int,
) -> dict:
    """Run one video through the full pipeline. Returns PitchData-compatible dict."""
    from backend.app.pipeline.smoothing import smooth_joints
    from backend.app.pipeline.features import extract_features
    from backend.app.pipeline.alignment import align_to_mound
    from backend.app.models.pitch import PitchMetadata, PitchData

    pitch_id = str(uuid.uuid4())
    print(f"  [{pitch_number}] {video_path.name} → {pitch_id[:8]}...")

    # Stage 1: Decode video to frames
    print("    Stage 1: video decode")
    frames = video_pipeline.decode(video_path)

    # Stage 2: SAM 3D Body inference → raw joints (T, 127, 3) and pose/shape params
    print("    Stage 2: SAM 3D Body inference")
    raw_joints, pose_params, shape_params = sam_pipeline.run(frames)

    # Stage 3: Temporal smoothing
    print("    Stage 3: joint smoothing")
    joints = smooth_joints(raw_joints)

    # Stage 4: MHR70 reduction
    print("    Stage 4: MHR70 reduction")
    joints_mhr70 = joints[:, :70, :]

    # Stage 5: Mound alignment (normalize coordinate frame)
    print("    Stage 5: mound alignment")
    joints = align_to_mound(joints)

    # Stage 6: Feature extraction (used downstream by analysis modules)
    print("    Stage 6: feature extraction")
    _features = extract_features(joints, pose_params)

    metadata = PitchMetadata(
        pitch_id=pitch_id,
        pitcher_id=pitcher_id,
        game_date=datetime.now(),
        inning=1,
        pitch_number=pitch_number,
        pitch_type=pitch_type,
        video_path=str(video_path),
        frame_start=0,
        frame_end=len(joints) - 1,
    )

    pitch_data = PitchData.from_numpy(
        metadata=metadata,
        joints=joints,
        pose_params=pose_params,
        shape_params=shape_params,
        joints_mhr70=joints_mhr70,
    )

    return pitch_data.model_dump()


def run_analysis(pitch_data_dict: dict, output_dir: Path) -> None:
    """Run all 6 analysis modules and save results."""
    from backend.app.analysis.baseline import BaselineComparison
    from backend.app.analysis.tipping import TippingDetection
    from backend.app.analysis.fatigue import FatigueTracking
    from backend.app.analysis.command import CommandAnalysis
    from backend.app.analysis.arm_slot import ArmSlotDrift
    from backend.app.analysis.timing import TimingAnalysis
    from backend.app.models.storage import StorageLayer

    pitcher_id = pitch_data_dict["metadata"]["pitcher_id"]
    pitch_id = pitch_data_dict["metadata"]["pitch_id"]

    storage = StorageLayer(
        db_path=output_dir / "baseball.db",
        parquet_dir=output_dir / "parquet",
    )

    baseline = storage.load_baseline(pitcher_id)
    modules = [
        BaselineComparison(storage=storage),
        TippingDetection(storage=storage),
        FatigueTracking(storage=storage),
        CommandAnalysis(storage=storage),
        ArmSlotDrift(storage=storage),
        TimingAnalysis(storage=storage),
    ]

    results = []
    for module in modules:
        try:
            result = module.analyze(pitch_data_dict, baseline=baseline)
            results.append(result.model_dump())
        except Exception as exc:
            print(f"    WARNING: {module.__class__.__name__} failed: {exc}")

    analysis_dir = output_dir / pitcher_id / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / f"{pitch_id}.json").write_text(json.dumps(results))


def main() -> None:
    args = parse_args()
    check_gpu_deps()

    video_files = sorted(args.input_dir.glob("*.mp4"))
    if not video_files:
        print(f"No MP4 files found in {args.input_dir}")
        sys.exit(1)

    print(f"SamPlaysBaseball — sample video processor")
    print(f"  Pitcher: {args.pitcher_name} ({args.pitcher_id}, {args.handedness}H)")
    print(f"  Videos:  {len(video_files)} MP4s in {args.input_dir}")
    print(f"  Output:  {args.output_dir}")
    print(f"  Device:  {args.device}")
    print()

    video_pipeline, sam_pipeline = load_pipeline(args.device)

    pitcher_dir = args.output_dir / args.pitcher_id
    pitcher_dir.mkdir(parents=True, exist_ok=True)

    all_joints = []  # Collect for baseline computation
    for idx, video_path in enumerate(video_files):
        pitch_data = process_video(
            video_path=video_path,
            pitcher_id=args.pitcher_id,
            pitch_type=args.pitch_type,
            pitch_number=idx + 1,
            video_pipeline=video_pipeline,
            sam_pipeline=sam_pipeline,
            fps=args.fps,
        )

        # Save pitch JSON
        pitch_id = pitch_data["metadata"]["pitch_id"]
        (pitcher_dir / f"{pitch_id}.json").write_text(json.dumps(pitch_data))

        # Run analysis
        run_analysis(pitch_data, args.output_dir)
        all_joints.append(pitch_data["joints"])

    print()
    print(f"Processed {len(video_files)} pitches.")
    print(f"Output written to: {args.output_dir / args.pitcher_id}")
    print()
    print("Next: run generate_synthetic.py to fill in any missing pitchers, then:")
    print("  python demo/launcher.py")


if __name__ == "__main__":
    main()
