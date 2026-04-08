"""Quick visual sanity check for a SAM 3D Body inference output.

Renders three frames (start, mid-delivery, follow-through) of a mesh .npz as
matplotlib 2D projections, side view + front view + top view. Saves PNGs to
data/clips/<game_pk>/_inspect_<play_id>.png so they can be opened with Read.

Usage:
    python scripts/inspect_mesh_quick.py <play_id>
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.app.data.pitch_db import PitchDB


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python scripts/inspect_mesh_quick.py <play_id>")
        sys.exit(1)
    play_id = sys.argv[1]

    db = PitchDB("data/pitches.db", "data/meshes")
    pitch = db.get_pitch(play_id)
    if not pitch or not pitch["mesh_path"]:
        print(f"No mesh for play_id={play_id}")
        sys.exit(1)

    data = np.load(pitch["mesh_path"])
    verts = data["vertices"]            # (frames, 18439, 3)
    joints = data["joints_mhr70"]       # (frames, 70, 3)
    n_frames = verts.shape[0]

    # Pick three frames: start, mid (likely windup/leg lift), late (release/follow-through)
    frame_indices = [0, n_frames // 2, n_frames - 10]
    titles = [f"frame {f}/{n_frames}" for f in frame_indices]

    fig, axes = plt.subplots(3, 3, figsize=(13, 12))

    for col, fi in enumerate(frame_indices):
        v = verts[fi]
        j = joints[fi]

        # Side view (Y up, X right) — XY plane
        ax = axes[0, col]
        ax.scatter(v[:, 0], v[:, 1], s=0.3, c="steelblue", alpha=0.4)
        ax.scatter(j[:, 0], j[:, 1], s=18, c="red", alpha=0.9)
        ax.set_title(f"SIDE — {titles[col]}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y (up)")
        ax.set_aspect("equal")
        ax.invert_yaxis()  # MHR convention: -Y is up

        # Front view (Y up, Z is depth from camera) — ZY plane
        ax = axes[1, col]
        ax.scatter(v[:, 2], v[:, 1], s=0.3, c="steelblue", alpha=0.4)
        ax.scatter(j[:, 2], j[:, 1], s=18, c="red", alpha=0.9)
        ax.set_title(f"FRONT (depth) — {titles[col]}")
        ax.set_xlabel("Z (depth)")
        ax.set_ylabel("Y (up)")
        ax.set_aspect("equal")
        ax.invert_yaxis()

        # Top view — XZ plane
        ax = axes[2, col]
        ax.scatter(v[:, 0], v[:, 2], s=0.3, c="steelblue", alpha=0.4)
        ax.scatter(j[:, 0], j[:, 2], s=18, c="red", alpha=0.9)
        ax.set_title(f"TOP — {titles[col]}")
        ax.set_xlabel("X")
        ax.set_ylabel("Z (depth)")
        ax.set_aspect("equal")

    fig.suptitle(f"Mesh inspection — {pitch['pitcher_name']} pitch {play_id[:8]}", fontsize=14)
    fig.tight_layout()

    out_dir = Path(f"data/clips/{pitch['game_pk']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"_inspect_{play_id[:8]}.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    print(f"WROTE: {out_path}")

    # Also dump joint trajectory diagnostics
    print()
    print(f"=== {pitch['pitcher_name']} {play_id[:8]} ===")
    print(f"frames: {n_frames}, fps: {data['source_fps']}")
    print(f"vertices: {verts.shape}  joints: {joints.shape}")
    # Body bbox
    bbox_diag = float(np.linalg.norm(verts[0].max(axis=0) - verts[0].min(axis=0)))
    print(f"body bbox diagonal at frame 0: {bbox_diag:.3f}")
    # Total body-center travel
    centers = verts.mean(axis=1)
    travel = float(np.linalg.norm(centers.max(axis=0) - centers.min(axis=0)))
    print(f"body-center travel across all frames: {travel:.3f}")
    # Joint sanity: head should be highest (smallest Y in MHR), feet lowest
    # Joint indices for MHR70 are model-specific; just check Y range
    y_min = float(joints[0, :, 1].min())
    y_max = float(joints[0, :, 1].max())
    print(f"joint Y range at frame 0: [{y_min:.3f}, {y_max:.3f}] (span={y_max-y_min:.3f})")
    db.close()


if __name__ == "__main__":
    main()
