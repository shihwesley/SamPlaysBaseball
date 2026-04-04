"""
Batch render multiple GLB files using Blender in headless mode.

Usage:
    python batch_render.py --input-dir ./glbs --output-dir ./renders --mode pitch --workers 4
"""
import sys
import os
import argparse
import subprocess
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

BLENDER_CMD = os.environ.get("BLENDER_PATH", "blender")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _run_blender(script: str, extra_args: list[str]) -> tuple[bool, str]:
    """Run a Blender script headless and return (success, output)."""
    cmd = [BLENDER_CMD, "--background", "--python", script, "--"] + extra_args
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        success = result.returncode == 0
        output = result.stdout + result.stderr
        return success, output
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT after 600s"
    except FileNotFoundError:
        return False, f"blender not found at '{BLENDER_CMD}'. Set BLENDER_PATH env var."


def render_one_pitch(args_tuple: tuple) -> bool:
    glb, output_dir, preset = args_tuple
    name = os.path.splitext(os.path.basename(glb))[0]
    out = os.path.join(output_dir, name)
    script = os.path.join(SCRIPT_DIR, "render_pitch.py")
    success, msg = _run_blender(script, ["--glb", glb, "--output", out, "--preset", preset])
    status = "OK" if success else "FAILED"
    print(f"[pitch] {name}: {status}")
    if not success:
        print(f"  {msg[:200]}")
    return success


def render_one_comparison(args_tuple: tuple) -> bool:
    glb1, glb2, output_dir, preset = args_tuple
    n1 = os.path.splitext(os.path.basename(glb1))[0]
    n2 = os.path.splitext(os.path.basename(glb2))[0]
    name = f"{n1}_vs_{n2}"
    out = os.path.join(output_dir, name)
    script = os.path.join(SCRIPT_DIR, "render_comparison.py")
    success, msg = _run_blender(
        script,
        ["--glb1", glb1, "--glb2", glb2, "--output", out, "--preset", preset, "--mode", "ghost"],
    )
    status = "OK" if success else "FAILED"
    print(f"[compare] {name}: {status}")
    if not success:
        print(f"  {msg[:200]}")
    return success


def render_one_strobe(args_tuple: tuple) -> bool:
    glbs, output_dir, idx = args_tuple
    out = os.path.join(output_dir, f"strobe_group_{idx:03d}")
    script = os.path.join(SCRIPT_DIR, "render_stroboscope.py")
    success, msg = _run_blender(script, ["--glbs"] + glbs + ["--output", out])
    status = "OK" if success else "FAILED"
    print(f"[strobe] group {idx} ({len(glbs)} pitches): {status}")
    if not success:
        print(f"  {msg[:200]}")
    return success


def main():
    parser = argparse.ArgumentParser(description="Batch render GLB files via Blender")
    parser.add_argument("--input-dir", required=True, help="Directory containing .glb files")
    parser.add_argument("--output-dir", required=True, help="Output directory for renders")
    parser.add_argument(
        "--mode",
        choices=["pitch", "comparison", "stroboscope"],
        default="pitch",
        help="Render mode",
    )
    parser.add_argument("--preset", default="catcher", help="Camera preset")
    parser.add_argument("--workers", type=int, default=2, help="Parallel Blender processes")
    parser.add_argument(
        "--group-size", type=int, default=20, help="Max pitches per stroboscope group"
    )
    args = parser.parse_args()

    glbs = sorted(glob.glob(os.path.join(args.input_dir, "*.glb")))
    if not glbs:
        print(f"No .glb files found in {args.input_dir}")
        sys.exit(1)

    print(f"Found {len(glbs)} GLB files in {args.input_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "pitch":
        tasks = [(g, args.output_dir, args.preset) for g in glbs]
        fn = render_one_pitch
    elif args.mode == "comparison":
        # Pair consecutive GLBs
        tasks = [
            (glbs[i], glbs[i + 1], args.output_dir, args.preset)
            for i in range(0, len(glbs) - 1, 2)
        ]
        fn = render_one_comparison
    else:  # stroboscope
        tasks = [
            (glbs[i: i + args.group_size], args.output_dir, i // args.group_size)
            for i in range(0, len(glbs), args.group_size)
        ]
        fn = render_one_strobe

    if not tasks:
        print("No tasks to render.")
        sys.exit(0)

    print(f"Running {len(tasks)} render tasks with {args.workers} workers...")

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fn, t): t for t in tasks}
        for future in as_completed(futures):
            results.append(future.result())

    ok = sum(results)
    total = len(results)
    print(f"\nBatch render complete: {ok}/{total} succeeded")
    if ok < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
