#!/usr/bin/env python3
"""Benchmark: FPS impact on 3D body mesh quality.

Runs inference on the same pitch clip at 60/30/24fps plus a variable-rate
strategy (30fps base + 60fps around release). Compares vertex positions
across all strategies against the 60fps baseline.

Usage:
    python -m benchmarks.bench_fps
    python -m benchmarks.bench_fps --clip data/clips/813024/inn2_ab11_p5_FF_55017c29.mp4
    python -m benchmarks.bench_fps --skip-inference  # reuse cached results

Output:
    - Per-strategy timing and frame counts
    - Vertex deviation stats (mean/max/P95 mm) vs 60fps baseline
    - Joint deviation stats
    - Per-phase breakdown (windup, stride, release, follow-through)
    - Recommendation based on results
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CACHE_DIR = PROJECT_ROOT / "data" / "benchmark_fps"
DEFAULT_CLIP = PROJECT_ROOT / "data" / "clips" / "813024" / "inn2_ab11_p5_FF_55017c29.mp4"
WEIGHTS_DIR = "/tmp/sam3d-mlx-weights/"


# ─── Frame extraction ────────────────────────────────────────────────────────

def _probe_video(video_path: str) -> tuple[int, int]:
    """Return (width, height) via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json", str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)["streams"][0]
    return int(info["width"]), int(info["height"])


def extract_frames_streaming(video_path: str, target_fps: int,
                             keep_indices: set[int] | None = None,
                             start_s: float = 0, duration_s: float = 0) -> list[np.ndarray]:
    """
    Extract frames at target FPS using streaming FFmpeg pipe.

    Reads one frame at a time from FFmpeg stdout to avoid buffering
    the entire video in memory.

    keep_indices: if provided, only keep frames at these indices.
    start_s: start time in seconds (0 = beginning).
    duration_s: duration in seconds (0 = full video).
    """
    w, h = _probe_video(video_path)
    frame_size = w * h * 3

    cmd = ["ffmpeg"]
    if start_s > 0:
        cmd += ["-ss", str(start_s)]
    cmd += ["-i", str(video_path)]
    if duration_s > 0:
        cmd += ["-t", str(duration_s)]
    cmd += [
        "-vf", f"fps={target_fps}",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    frames = []
    idx = 0
    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break
        if keep_indices is None or idx in keep_indices:
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()
            frames.append(frame)
        idx += 1

    proc.stdout.close()
    proc.wait()
    return frames


def count_frames_at_fps(video_path: str, target_fps: int,
                        start_s: float = 0, duration_s: float = 0) -> int:
    """Count frames by streaming (no buffering)."""
    w, h = _probe_video(video_path)
    frame_size = w * h * 3

    cmd = ["ffmpeg"]
    if start_s > 0:
        cmd += ["-ss", str(start_s)]
    cmd += ["-i", str(video_path)]
    if duration_s > 0:
        cmd += ["-t", str(duration_s)]
    cmd += ["-vf", f"fps={target_fps}", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    count = 0
    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break
        count += 1
    proc.stdout.close()
    proc.wait()
    return count


def build_variable_indices(n_total: int) -> tuple[set[int], list[int]]:
    """
    Compute which 60fps frame indices to keep for variable-rate strategy.

    30fps outside the release window, 60fps inside it.
    Returns (keep_set, ordered_list).
    """
    release_start = int(n_total * 0.55)
    release_end = int(n_total * 0.80)

    indices = []
    for i in range(n_total):
        if release_start <= i < release_end:
            indices.append(i)
        elif i % 2 == 0:
            indices.append(i)

    return set(indices), indices


# ─── Inference ────────────────────────────────────────────────────────────────

def run_inference(frames: list[np.ndarray], label: str,
                  weights_dir: str = WEIGHTS_DIR) -> dict:
    """Run SAM 3D Body inference on frames, return vertices + timing."""
    from sam3d_mlx.estimator import SAM3DBodyEstimator, detect_persons_cached

    estimator = SAM3DBodyEstimator(weights_dir)
    # Warm up
    detect_persons_cached(np.zeros((100, 100, 3), dtype=np.uint8))
    _ = estimator.predict(frames[0], [0, 0, frames[0].shape[1], frames[0].shape[0]])

    all_vertices = []
    all_joints = []
    frame_times = []
    detection_cache = {}

    print(f"\n  [{label}] Running inference on {len(frames)} frames...")

    for i, frame in enumerate(frames):
        t0 = time.time()

        h, w = frame.shape[:2]
        detections = detect_persons_cached(frame, threshold=0.5)
        bbox = detections[0] if detections else [0, 0, w, h]

        result = estimator.predict(frame, bbox, auto_detect=False)

        elapsed = time.time() - t0
        frame_times.append(elapsed)

        all_vertices.append(result["pred_vertices"])
        all_joints.append(result["pred_keypoints_3d"])

        if (i + 1) % 20 == 0 or i == len(frames) - 1:
            avg_ms = np.mean(frame_times[-20:]) * 1000
            print(f"    Frame {i+1}/{len(frames)} — avg {avg_ms:.0f}ms/frame")

    vertices = np.stack(all_vertices, axis=0)   # (T, V, 3)
    joints = np.stack(all_joints, axis=0)        # (T, J, 3)

    total_time = sum(frame_times)
    return {
        "vertices": vertices,
        "joints": joints,
        "frame_times": np.array(frame_times),
        "total_time": total_time,
        "n_frames": len(frames),
    }


# ─── Comparison ───────────────────────────────────────────────────────────────

def interpolate_to_60fps(vertices_low: np.ndarray, joints_low: np.ndarray,
                         n_target: int, source_fps: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Linearly interpolate lower-FPS results back to 60fps timeline
    so we can compare vertex positions frame-by-frame.
    """
    n_low = vertices_low.shape[0]
    # Map each 60fps frame index to a position in the low-fps array
    low_times = np.arange(n_low) / source_fps
    high_times = np.arange(n_target) / 60.0

    # Clip to the range covered by the low-fps data
    max_time = low_times[-1]
    high_times = np.clip(high_times, 0, max_time)

    # Per-vertex linear interpolation
    n_verts = vertices_low.shape[1]
    interp_verts = np.zeros((n_target, n_verts, 3), dtype=np.float32)
    n_joints = joints_low.shape[1]
    interp_joints = np.zeros((n_target, n_joints, 3), dtype=np.float32)

    for t_idx, t in enumerate(high_times):
        # Find bracketing frames in low-fps
        frac_idx = t * source_fps
        lo = int(np.floor(frac_idx))
        hi = min(lo + 1, n_low - 1)
        alpha = frac_idx - lo

        interp_verts[t_idx] = (1 - alpha) * vertices_low[lo] + alpha * vertices_low[hi]
        interp_joints[t_idx] = (1 - alpha) * joints_low[lo] + alpha * joints_low[hi]

    return interp_verts, interp_joints


def interpolate_variable_to_60fps(vertices: np.ndarray, joints: np.ndarray,
                                   indices: list[int], n_target: int) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate variable-rate results back to 60fps using original indices."""
    n_verts = vertices.shape[1]
    n_joints = joints.shape[1]
    interp_verts = np.zeros((n_target, n_verts, 3), dtype=np.float32)
    interp_joints = np.zeros((n_target, n_joints, 3), dtype=np.float32)

    for t in range(n_target):
        # Find bracketing indices
        lo_pos = 0
        for j, idx in enumerate(indices):
            if idx <= t:
                lo_pos = j
            else:
                break

        hi_pos = min(lo_pos + 1, len(indices) - 1)
        lo_idx = indices[lo_pos]
        hi_idx = indices[hi_pos]

        if hi_idx == lo_idx:
            alpha = 0.0
        else:
            alpha = (t - lo_idx) / (hi_idx - lo_idx)

        interp_verts[t] = (1 - alpha) * vertices[lo_pos] + alpha * vertices[hi_pos]
        interp_joints[t] = (1 - alpha) * joints[lo_pos] + alpha * joints[hi_pos]

    return interp_verts, interp_joints


def compute_deviation(baseline: np.ndarray, candidate: np.ndarray) -> dict:
    """
    Compare two (T, V, 3) arrays. Returns per-frame Euclidean distance stats.
    All values in millimeters.
    """
    n_frames = min(baseline.shape[0], candidate.shape[0])
    base = baseline[:n_frames]
    cand = candidate[:n_frames]

    # Per-vertex Euclidean distance per frame
    per_vertex_dist = np.linalg.norm(base - cand, axis=-1)  # (T, V)
    per_vertex_mm = per_vertex_dist * 1000  # meters → mm

    # Per-frame stats
    frame_mean = per_vertex_mm.mean(axis=1)   # (T,)
    frame_max = per_vertex_mm.max(axis=1)     # (T,)

    # Phase breakdown (approximate)
    phase_pcts = {
        "windup": (0.0, 0.35),
        "stride": (0.35, 0.55),
        "arm_cocking": (0.55, 0.70),
        "release": (0.70, 0.82),
        "follow_through": (0.82, 1.0),
    }
    phase_stats = {}
    for phase, (start_pct, end_pct) in phase_pcts.items():
        s = int(n_frames * start_pct)
        e = int(n_frames * end_pct)
        if e > s:
            phase_stats[phase] = {
                "mean_mm": float(frame_mean[s:e].mean()),
                "max_mm": float(frame_max[s:e].max()),
                "p95_mm": float(np.percentile(per_vertex_mm[s:e], 95)),
            }

    return {
        "mean_mm": float(per_vertex_mm.mean()),
        "max_mm": float(per_vertex_mm.max()),
        "p95_mm": float(np.percentile(per_vertex_mm, 95)),
        "p99_mm": float(np.percentile(per_vertex_mm, 99)),
        "median_mm": float(np.median(per_vertex_mm)),
        "frame_mean": frame_mean,
        "frame_max": frame_max,
        "phases": phase_stats,
    }


# ─── Report ───────────────────────────────────────────────────────────────────

def print_report(results: dict):
    """Print comparison report."""
    print("\n" + "=" * 70)
    print("  FPS BENCHMARK RESULTS")
    print("=" * 70)

    baseline = results["60fps"]

    # Timing summary
    print("\n── Timing ──────────────────────────────────────────────────────")
    print(f"{'Strategy':<15} {'Frames':<8} {'Total (s)':<12} {'Per-frame (ms)':<15} {'Speedup':<8}")
    print("-" * 60)
    for name, r in results.items():
        fps_label = name
        n = r["n_frames"]
        total = r["total_time"]
        per_frame = np.median(r["frame_times"]) * 1000
        speedup = baseline["total_time"] / total if total > 0 else 0
        print(f"{fps_label:<15} {n:<8} {total:<12.1f} {per_frame:<15.0f} {speedup:<8.2f}x")

    # Quality comparison
    print("\n── Vertex Deviation vs 60fps Baseline ──────────────────────────")
    print(f"{'Strategy':<15} {'Mean (mm)':<12} {'Median (mm)':<13} {'P95 (mm)':<12} {'Max (mm)':<12}")
    print("-" * 65)
    print(f"{'60fps':<15} {'0.000':<12} {'0.000':<13} {'0.000':<12} {'0.000':<12}")

    deviations = {}
    for name, r in results.items():
        if name == "60fps":
            continue
        dev = r["vertex_deviation"]
        deviations[name] = dev
        print(f"{name:<15} {dev['mean_mm']:<12.3f} {dev['median_mm']:<13.3f} "
              f"{dev['p95_mm']:<12.3f} {dev['max_mm']:<12.3f}")

    # Joint deviation
    print("\n── Joint Deviation vs 60fps Baseline ───────────────────────────")
    print(f"{'Strategy':<15} {'Mean (mm)':<12} {'P95 (mm)':<12} {'Max (mm)':<12}")
    print("-" * 50)
    for name, r in results.items():
        if name == "60fps":
            continue
        dev = r["joint_deviation"]
        print(f"{name:<15} {dev['mean_mm']:<12.3f} {dev['p95_mm']:<12.3f} {dev['max_mm']:<12.3f}")

    # Phase breakdown
    print("\n── Phase Breakdown (Mean Vertex Deviation mm) ──────────────────")
    phases = ["windup", "stride", "arm_cocking", "release", "follow_through"]
    header = f"{'Strategy':<15}" + "".join(f"{p:<15}" for p in phases)
    print(header)
    print("-" * (15 + 15 * len(phases)))
    for name, dev in deviations.items():
        row = f"{name:<15}"
        for phase in phases:
            if phase in dev["phases"]:
                row += f"{dev['phases'][phase]['mean_mm']:<15.3f}"
            else:
                row += f"{'—':<15}"
        print(row)

    # Recommendation
    print("\n── Recommendation ──────────────────────────────────────────────")

    best = None
    for name in ["24fps", "30fps", "variable"]:
        if name not in deviations:
            continue
        dev = deviations[name]
        # "Acceptable" = mean deviation < 2mm, P95 < 5mm
        if dev["mean_mm"] < 2.0 and dev["p95_mm"] < 5.0:
            if best is None:
                best = name
            elif results[name]["total_time"] < results[best]["total_time"]:
                best = name

    if best:
        speedup = baseline["total_time"] / results[best]["total_time"]
        dev = deviations[best]
        print(f"  {best} is the sweet spot:")
        print(f"    - {speedup:.1f}x faster than 60fps")
        print(f"    - {dev['mean_mm']:.2f}mm mean deviation ({dev['p95_mm']:.2f}mm P95)")
        print(f"    - {results[best]['n_frames']} frames vs {baseline['n_frames']} at 60fps")

        # Check release phase specifically
        if "release" in dev["phases"]:
            rel = dev["phases"]["release"]
            print(f"    - Release phase: {rel['mean_mm']:.2f}mm mean, {rel['max_mm']:.2f}mm max")
            if rel["mean_mm"] > 3.0:
                print(f"    ⚠ Release phase deviation is elevated — consider variable-rate")
    else:
        print("  All strategies show >2mm mean deviation. 60fps recommended.")
        print("  (This may indicate the model is sensitive to temporal density,")
        print("  or the clip has fast motion that benefits from higher sampling.)")

    print("\n" + "=" * 70)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark FPS impact on mesh quality")
    parser.add_argument("--clip", default=str(DEFAULT_CLIP), help="Input video clip")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Reuse cached results from previous run")
    parser.add_argument("--weights", default=WEIGHTS_DIR, help="Model weights directory")
    parser.add_argument("--save-meshes", action="store_true",
                        help="Save per-strategy vertex arrays for inspection")
    parser.add_argument("--only", choices=["60fps", "30fps", "24fps", "variable"],
                        help="Run only one strategy (for sequential execution)")
    parser.add_argument("--start", type=float, default=1.0,
                        help="Start time in seconds (default: 1.0)")
    parser.add_argument("--duration", type=float, default=3.0,
                        help="Duration in seconds (default: 3.0)")
    args = parser.parse_args()

    _weights = args.weights
    _start = args.start
    _dur = args.duration

    clip = args.clip
    if not os.path.exists(clip):
        print(f"Error: clip not found: {clip}")
        sys.exit(1)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    strategies = ["60fps", "30fps", "24fps", "variable"]
    run_strategies = [args.only] if args.only else strategies

    print("=" * 70)
    print("  FPS BENCHMARK: " + " vs ".join(run_strategies))
    print(f"  Clip: {clip}")
    print(f"  Window: {_start}s – {_start + _dur}s ({_dur}s)")
    print("=" * 70)

    if args.skip_inference:
        print("\nLoading cached results...")
    else:
        # Count 60fps frames for variable strategy planning
        n_60 = count_frames_at_fps(clip, 60, start_s=_start, duration_s=_dur)
        _, var_indices = build_variable_indices(n_60)

        # Save var_indices for later
        np.save(CACHE_DIR / "var_indices.npy", np.array(var_indices))

        fps_map = {"60fps": 60, "30fps": 30, "24fps": 24}

        for strategy in run_strategies:
            print(f"\n── {strategy} ─────────────────────────────────────────────")
            strat_file = CACHE_DIR / f"{strategy}.npz"

            # Extract frames
            print(f"  Extracting frames...", end=" ", flush=True)
            if strategy == "variable":
                # Extract 60fps and select variable subset
                all_60 = extract_frames_streaming(clip, 60, start_s=_start, duration_s=_dur)
                frames = [all_60[i] for i in var_indices if i < len(all_60)]
                del all_60  # free memory
                print(f"{len(frames)} frames (variable rate from {n_60})")
            else:
                fps = fps_map[strategy]
                frames = extract_frames_streaming(clip, fps, start_s=_start, duration_s=_dur)
                print(f"{len(frames)} frames")

            # Run inference
            result = run_inference(frames, strategy, weights_dir=_weights)

            # Save to disk immediately, free memory
            np.savez_compressed(
                strat_file,
                vertices=result["vertices"],
                joints=result["joints"],
                frame_times=result["frame_times"],
                total_time=np.array(result["total_time"]),
                n_frames=np.array(result["n_frames"]),
            )
            print(f"  Saved to {strat_file}")
            del frames, result  # free memory

    # Load all results back from disk for comparison
    results = {}
    var_indices = list(np.load(CACHE_DIR / "var_indices.npy"))

    for strategy in strategies:
        strat_file = CACHE_DIR / f"{strategy}.npz"
        if not strat_file.exists():
            if strategy in run_strategies:
                print(f"  Warning: {strat_file} not found")
            continue
        data = np.load(strat_file)
        results[strategy] = {
            "vertices": data["vertices"],
            "joints": data["joints"],
            "frame_times": data["frame_times"],
            "total_time": float(data["total_time"]),
            "n_frames": int(data["n_frames"]),
        }

    if "60fps" not in results:
        print("\nError: 60fps baseline not found. Run with --only 60fps first.")
        sys.exit(1)

    if len(results) < 2:
        print(f"\nOnly {list(results.keys())} available. Run remaining strategies to compare.")
        sys.exit(0)

    # Compute deviations
    print("\n── Computing deviations ────────────────────────────────────")
    baseline_verts = results["60fps"]["vertices"]
    baseline_joints = results["60fps"]["joints"]
    n_baseline = baseline_verts.shape[0]

    fps_rates = {"30fps": 30, "24fps": 24}
    for name, fps in fps_rates.items():
        if name not in results:
            continue
        interp_v, interp_j = interpolate_to_60fps(
            results[name]["vertices"], results[name]["joints"], n_baseline, fps)
        results[name]["vertex_deviation"] = compute_deviation(baseline_verts, interp_v)
        results[name]["joint_deviation"] = compute_deviation(baseline_joints, interp_j)

    if "variable" in results:
        interp_var_v, interp_var_j = interpolate_variable_to_60fps(
            results["variable"]["vertices"], results["variable"]["joints"],
            var_indices, n_baseline)
        results["variable"]["vertex_deviation"] = compute_deviation(baseline_verts, interp_var_v)
        results["variable"]["joint_deviation"] = compute_deviation(baseline_joints, interp_var_j)

    # Save mesh arrays if requested
    if args.save_meshes:
        for name, r in results.items():
            out = CACHE_DIR / f"vertices_{name}.npy"
            np.save(out, r["vertices"])
            print(f"  Saved {out}")

    print_report(results)


if __name__ == "__main__":
    main()
