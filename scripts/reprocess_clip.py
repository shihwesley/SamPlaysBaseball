"""Re-run inference on a single clip with the new bbox-tracking + fps capture.

Writes the resulting MeshData to a chosen output path *without touching the
SQLite database*. Lets you compare an original .npz vs a freshly-processed
one side-by-side in Blender:

    SAMPLAYS_MESH_PATH=/path/to/old.npz blender --python scripts/blender_field_viewer.py
    SAMPLAYS_MESH_PATH=/path/to/new.npz blender --python scripts/blender_field_viewer.py

Usage:
    python scripts/reprocess_clip.py \\
        --video data/clips/813024/inn1_ab5_p1_CU_02ec65f0.mp4 \\
        --output data/meshes/813024/02ec65f0_fixed.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, "/tmp/sam-3d-body")

from scripts.batch_inference import load_model_mlx, process_clip_mlx  # noqa: E402
from backend.app.data.pitch_db import MeshData  # noqa: E402


def _save_mesh_npz(mesh: MeshData, output: Path) -> None:
    """Mirror PitchDB._save_mesh, but write to an arbitrary path."""
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(
        vertices=mesh.vertices,
        joints_mhr70=mesh.joints_mhr70,
        pose_params=mesh.pose_params,
        shape_params=mesh.shape_params,
        cam_t=mesh.cam_t,
        focal_length=np.array(mesh.focal_length),
    )
    if mesh.faces is not None:
        arrays["faces"] = mesh.faces
    if mesh.source_fps is not None:
        arrays["source_fps"] = np.array(float(mesh.source_fps))

    # Backfill faces from the project cache so the Blender viewer can
    # render the result without needing to find data/faces.npy.
    if "faces" not in arrays:
        cached_faces = PROJECT_ROOT / "data" / "faces.npy"
        if cached_faces.exists():
            arrays["faces"] = np.load(cached_faces)

    np.savez_compressed(output, **arrays)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Path to input clip (.mp4)")
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--det-threshold", type=float, default=0.5)
    args = parser.parse_args()

    video_path = Path(args.video)
    output_path = Path(args.output)
    if not video_path.exists():
        print(f"Error: video not found: {video_path}")
        return 1

    print(f"Loading MLX model...")
    estimator = load_model_mlx()

    print(f"Processing {video_path.name}...")
    mesh, total_ms = process_clip_mlx(
        str(video_path), estimator, det_threshold=args.det_threshold,
    )
    if mesh is None:
        print("Inference produced no frames.")
        return 1

    _save_mesh_npz(mesh, output_path)
    n = mesh.vertices.shape[0]
    fps_str = f"{mesh.source_fps:.2f} fps" if mesh.source_fps else "no fps in npz"
    print()
    print(f"Wrote {output_path}")
    print(f"  {n} frames, {mesh.vertices.shape[1]} verts, {fps_str}")
    print(f"  Inference: {total_ms / 1000:.1f}s "
          f"({n / (total_ms / 1000):.1f} fps inference throughput)")
    print()
    print("Open in Blender:")
    print(f"  SAMPLAYS_MESH_PATH={output_path} \\")
    print(f"    /Applications/Blender.app/Contents/MacOS/Blender "
          f"--python scripts/blender_field_viewer.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
