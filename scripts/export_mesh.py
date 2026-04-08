#!/usr/bin/env python3
"""Export a pitch mesh from the database to .glb, .obj, or .npz-with-faces.

Usage:
    # Export as GLB (animated, for Blender viewer)
    python scripts/export_mesh.py --play-id 02ec65f0-9054-3c9c-a72a-aaa3c610f0c9 --format glb

    # Export single frame as OBJ
    python scripts/export_mesh.py --play-id 02ec65f0 --format obj --frame 0

    # Export all frames as OBJ sequence
    python scripts/export_mesh.py --play-id 02ec65f0 --format obj --all-frames

    # List available pitches
    python scripts/export_mesh.py --list

    # Backfill faces into existing .npz files (run once)
    python scripts/export_mesh.py --backfill-faces

Faces are loaded from the safetensors weight file (character.mesh.faces)
and cached to data/faces.npy for reuse without loading the full model.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FACES_CACHE = PROJECT_ROOT / "data" / "faces.npy"
DEFAULT_WEIGHTS = "/tmp/sam3d-mlx-weights/"
DEFAULT_DB = PROJECT_ROOT / "data" / "pitches.db"
DEFAULT_MESH_DIR = PROJECT_ROOT / "data" / "meshes"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "export"


def load_faces(weights_dir: str = DEFAULT_WEIGHTS) -> np.ndarray:
    """Load face indices, using cache if available."""
    if FACES_CACHE.exists():
        return np.load(FACES_CACHE)

    # Extract from safetensors
    safetensors_path = os.path.join(weights_dir, "model.safetensors")
    if not os.path.exists(safetensors_path):
        print(f"Error: weights not found at {safetensors_path}")
        print("Run with --weights to specify the weights directory.")
        sys.exit(1)

    from safetensors.numpy import safe_open
    with safe_open(safetensors_path, framework="numpy") as f:
        faces = f.get_tensor("head_pose.faces")

    # Cache for future use
    FACES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(FACES_CACHE, faces)
    print(f"Cached faces to {FACES_CACHE} ({faces.shape})")

    return faces


def export_glb(vertices: np.ndarray, faces: np.ndarray, output_path: str,
               metadata: dict | None = None) -> str:
    """Export full animation as GLB with morph targets."""
    from backend.app.export.glb import GLBExporter

    frames = [vertices[t] for t in range(vertices.shape[0])]
    meta = metadata or {
        "pitch_type": "",
        "pitcher_id": "",
        "frame_timestamps": list(range(len(frames))),
        "phase_markers": {},
    }

    exporter = GLBExporter(
        vertex_count=vertices.shape[1],
        face_count=faces.shape[0],
    )
    return exporter.export_pitch(frames, meta, output_path)


def export_obj(vertices: np.ndarray, faces: np.ndarray, output_path: str) -> str:
    """Export a single frame as OBJ."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    return output_path


def list_pitches(db_path: Path, mesh_dir: Path):
    """Print available pitches with mesh data."""
    from backend.app.data.pitch_db import PitchDB
    db = PitchDB(db_path=db_path, mesh_dir=mesh_dir)
    stats = db.stats()

    print(f"\nDatabase: {db_path}")
    print(f"Total pitches: {stats['total_pitches']}")
    print(f"With mesh: {stats['with_mesh']}")
    print()

    rows = db._conn.execute(
        "SELECT play_id, pitcher_name, pitch_type, release_speed, num_frames, mesh_path "
        "FROM pitches WHERE mesh_path IS NOT NULL "
        "ORDER BY game_date DESC LIMIT 20"
    ).fetchall()

    if not rows:
        print("No pitches with mesh data found.")
        return

    print(f"{'Play ID':<40} {'Pitcher':<20} {'Type':<5} {'MPH':<6} {'Frames':<7}")
    print("-" * 80)
    for r in rows:
        speed = f"{r['release_speed']:.1f}" if r['release_speed'] else "?"
        frames = r['num_frames'] or "?"
        print(f"{r['play_id']:<40} {r['pitcher_name'] or '?':<20} "
              f"{r['pitch_type'] or '?':<5} {speed:<6} {frames:<7}")


def backfill_faces(mesh_dir: Path, weights_dir: str):
    """Add faces array to all existing .npz files that lack it."""
    faces = load_faces(weights_dir)
    npz_files = list(mesh_dir.rglob("*.npz"))

    if not npz_files:
        print("No .npz files found.")
        return

    updated = 0
    for npz_path in npz_files:
        data = dict(np.load(npz_path))
        if "faces" in data:
            continue

        data["faces"] = faces
        np.savez_compressed(npz_path, **data)
        updated += 1
        print(f"  Updated: {npz_path}")

    print(f"\nBackfilled {updated}/{len(npz_files)} files.")


def main():
    parser = argparse.ArgumentParser(description="Export pitch meshes for Blender")
    parser.add_argument("--play-id", help="Pitch play_id (or prefix)")
    parser.add_argument("--format", choices=["glb", "obj"], default="glb",
                        help="Output format (default: glb)")
    parser.add_argument("--frame", type=int, default=0,
                        help="Frame index for OBJ export (default: 0)")
    parser.add_argument("--all-frames", action="store_true",
                        help="Export all frames as OBJ sequence")
    parser.add_argument("--output", "-o", help="Output path (auto-generated if omitted)")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS,
                        help="Weights directory for face extraction")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite database path")
    parser.add_argument("--mesh-dir", default=str(DEFAULT_MESH_DIR), help="Mesh directory")
    parser.add_argument("--list", action="store_true", help="List available pitches")
    parser.add_argument("--backfill-faces", action="store_true",
                        help="Add faces to existing .npz files")
    parser.add_argument("--from-npz", help="Export directly from an .npz file (skip DB)")
    args = parser.parse_args()

    if args.list:
        list_pitches(Path(args.db), Path(args.mesh_dir))
        return

    if args.backfill_faces:
        backfill_faces(Path(args.mesh_dir), args.weights)
        return

    # Load mesh data
    if args.from_npz:
        npz_path = Path(args.from_npz)
        if not npz_path.exists():
            print(f"Error: {npz_path} not found")
            sys.exit(1)
        data = np.load(npz_path)
        vertices = data["vertices"]
        faces = data["faces"] if "faces" in data else load_faces(args.weights)
        play_id = npz_path.stem
        metadata = None
    else:
        if not args.play_id:
            print("Error: --play-id or --from-npz required")
            parser.print_help()
            sys.exit(1)

        from backend.app.data.pitch_db import PitchDB
        db = PitchDB(db_path=args.db, mesh_dir=args.mesh_dir)

        # Support prefix matching
        play_id = args.play_id
        row = db._conn.execute(
            "SELECT play_id, pitcher_name, pitch_type, release_speed "
            "FROM pitches WHERE play_id LIKE ?",
            (f"{play_id}%",)
        ).fetchone()

        if not row:
            print(f"No pitch found matching '{play_id}'")
            sys.exit(1)

        full_id = row["play_id"]
        print(f"Pitch: {row['pitcher_name']} — {row['pitch_type']} "
              f"@ {row['release_speed']:.1f} mph")

        mesh = db.load_mesh(full_id)
        if mesh is None:
            print(f"No mesh data for {full_id}")
            sys.exit(1)

        vertices = mesh.vertices
        faces = mesh.faces if mesh.faces is not None else load_faces(args.weights)
        play_id = full_id[:8]
        metadata = {
            "pitch_type": row["pitch_type"] or "",
            "pitcher_id": str(row.get("pitcher_id", "")),
            "pitcher_name": row["pitcher_name"] or "",
            "frame_timestamps": list(range(vertices.shape[0])),
            "phase_markers": {},
        }

    print(f"Mesh: {vertices.shape[0]} frames, {vertices.shape[1]} vertices, "
          f"{faces.shape[0]} faces")

    # Set up output directory
    output_dir = Path(args.output).parent if args.output else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export
    if args.format == "glb":
        out_path = args.output or str(output_dir / f"{play_id}.glb")
        export_glb(vertices, faces, out_path, metadata)
        print(f"Exported GLB: {out_path}")
        print(f"  Open in Blender: set MESH_PATH in blender_field_viewer.py")

    elif args.format == "obj":
        if args.all_frames:
            for t in range(vertices.shape[0]):
                frame_path = str(output_dir / f"{play_id}_frame{t:03d}.obj")
                export_obj(vertices[t], faces, frame_path)
            print(f"Exported {vertices.shape[0]} OBJ frames to {output_dir}/")
        else:
            frame = min(args.frame, vertices.shape[0] - 1)
            out_path = args.output or str(output_dir / f"{play_id}_frame{frame:03d}.obj")
            export_obj(vertices[frame], faces, out_path)
            print(f"Exported OBJ (frame {frame}): {out_path}")


if __name__ == "__main__":
    main()
