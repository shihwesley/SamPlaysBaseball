#!/usr/bin/env python3
"""Render benchmark results as side-by-side comparison videos.

Produces a 2x2 grid video showing the mesh at 60/30/24/variable fps,
all playing at the same wall-clock speed so you can see the differences.

Usage:
    python benchmarks/render_fps_comparison.py
    python benchmarks/render_fps_comparison.py --output comparison.mp4
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CACHE_DIR = PROJECT_ROOT / "data" / "benchmark_fps"
FACES_CACHE = PROJECT_ROOT / "data" / "faces.npy"
WEIGHTS_DIR = "/tmp/sam3d-mlx-weights/"


def load_faces() -> np.ndarray:
    """Load face indices from cache or extract from weights."""
    if FACES_CACHE.exists():
        return np.load(FACES_CACHE)
    from safetensors.numpy import safe_open
    with safe_open(f"{WEIGHTS_DIR}/model.safetensors", framework="numpy") as f:
        faces = f.get_tensor("head_pose.faces")
    FACES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(FACES_CACHE, faces)
    return faces


def interpolate_to_target(vertices: np.ndarray, source_fps: int,
                          target_n: int, target_fps: int = 60) -> np.ndarray:
    """Interpolate vertices from source_fps to target frame count."""
    n_src = vertices.shape[0]
    src_times = np.arange(n_src) / source_fps
    tgt_times = np.arange(target_n) / target_fps
    max_time = src_times[-1]
    tgt_times = np.clip(tgt_times, 0, max_time)

    result = np.zeros((target_n, vertices.shape[1], 3), dtype=np.float32)
    for t_idx, t in enumerate(tgt_times):
        frac = t * source_fps
        lo = int(np.floor(frac))
        hi = min(lo + 1, n_src - 1)
        alpha = frac - lo
        result[t_idx] = (1 - alpha) * vertices[lo] + alpha * vertices[hi]
    return result


def interpolate_variable(vertices: np.ndarray, indices: list[int],
                         target_n: int) -> np.ndarray:
    """Interpolate variable-rate vertices back to 60fps timeline."""
    result = np.zeros((target_n, vertices.shape[1], 3), dtype=np.float32)
    for t in range(target_n):
        lo_pos = 0
        for j, idx in enumerate(indices):
            if idx <= t:
                lo_pos = j
            else:
                break
        hi_pos = min(lo_pos + 1, len(indices) - 1)
        lo_idx = indices[lo_pos]
        hi_idx = indices[hi_pos]
        alpha = (t - lo_idx) / (hi_idx - lo_idx) if hi_idx != lo_idx else 0.0
        result[t] = (1 - alpha) * vertices[lo_pos] + alpha * vertices[hi_pos]
    return result


def render_mesh_frame(vertices: np.ndarray, faces: np.ndarray,
                      width: int = 480, height: int = 480) -> np.ndarray:
    """Render a single mesh frame using pyrender."""
    import trimesh
    import pyrender

    mesh = trimesh.Trimesh(vertices.copy(), faces.copy())
    # Flip for camera convention
    mesh.apply_transform(trimesh.transformations.rotation_matrix(
        np.radians(180), [1, 0, 0]))

    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0, alphaMode="OPAQUE",
        baseColorFactor=(0.65, 0.74, 0.86, 1.0))
    py_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

    scene = pyrender.Scene(bg_color=[0.1, 0.1, 0.1, 1.0],
                           ambient_light=(0.3, 0.3, 0.3))
    scene.add(py_mesh)

    # Camera: look at mesh center
    center = vertices.mean(axis=0)
    cam_t = center.copy()
    cam_t[0] *= -1
    cam_t[2] = abs(cam_t[2]) + 0.5  # push camera back

    camera_pose = np.eye(4)
    camera_pose[:3, 3] = cam_t
    focal = max(width, height) * 1.2
    scene.add(pyrender.IntrinsicsCamera(
        fx=focal, fy=focal, cx=width / 2, cy=height / 2, zfar=1e12),
        pose=camera_pose)

    # Directional light
    light_pose = np.eye(4)
    light_pose[:3, 3] = [0, 2, 2]
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.0),
              pose=light_pose)

    renderer = pyrender.OffscreenRenderer(viewport_width=width,
                                          viewport_height=height)
    color, _ = renderer.render(scene)
    renderer.delete()
    return color


def add_label(frame: np.ndarray, text: str) -> np.ndarray:
    """Add a text label to the top-left of a frame."""
    import cv2
    frame = frame.copy()
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Render FPS benchmark comparison")
    parser.add_argument("--output", "-o",
                        default=str(CACHE_DIR / "fps_comparison.mp4"))
    parser.add_argument("--fps", type=int, default=30,
                        help="Output video FPS (default: 30)")
    parser.add_argument("--size", type=int, default=480,
                        help="Per-panel size in pixels (default: 480)")
    parser.add_argument("--max-frames", type=int, default=90,
                        help="Max frames to render (default: 90 = 3s at 30fps)")
    args = parser.parse_args()

    import cv2

    # Load benchmark results
    print("Loading benchmark results...")
    strategies = {}
    for name in ["60fps", "30fps", "24fps", "variable"]:
        f = CACHE_DIR / f"{name}.npz"
        if f.exists():
            strategies[name] = np.load(f)
            print(f"  {name}: {strategies[name]['vertices'].shape}")

    if "60fps" not in strategies:
        print("Error: 60fps baseline not found")
        sys.exit(1)

    var_indices = list(np.load(CACHE_DIR / "var_indices.npy"))
    faces = load_faces()

    # Interpolate all to 60fps timeline
    n_baseline = strategies["60fps"]["vertices"].shape[0]
    n_render = min(n_baseline, args.max_frames * 2)  # 60fps source, render at 30fps

    print(f"\nInterpolating to {n_render} frames...")
    aligned = {}
    aligned["60fps"] = strategies["60fps"]["vertices"][:n_render]

    if "30fps" in strategies:
        aligned["30fps"] = interpolate_to_target(
            strategies["30fps"]["vertices"], 30, n_render, 60)

    if "24fps" in strategies:
        aligned["24fps"] = interpolate_to_target(
            strategies["24fps"]["vertices"], 24, n_render, 60)

    if "variable" in strategies:
        aligned["variable"] = interpolate_variable(
            strategies["variable"]["vertices"], var_indices, n_render)

    # Render comparison video
    panel_w = args.size
    panel_h = args.size
    grid_w = panel_w * 2
    grid_h = panel_h * 2

    out_path = args.output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, args.fps, (grid_w, grid_h))

    labels = ["60fps (baseline)", "30fps", "24fps", "variable"]
    keys = ["60fps", "30fps", "24fps", "variable"]

    # Render every other frame (60fps source → 30fps output)
    render_indices = list(range(0, n_render, 2))
    total = len(render_indices)

    print(f"Rendering {total} frames as 2x2 grid ({grid_w}x{grid_h})...")

    for render_idx, frame_idx in enumerate(render_indices):
        panels = []
        for key, label in zip(keys, labels):
            if key in aligned:
                verts = aligned[key][frame_idx]
                panel = render_mesh_frame(verts, faces, panel_w, panel_h)
                panel = add_label(panel, label)
            else:
                panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
                panel = add_label(panel, f"{label} (missing)")
            panels.append(panel)

        # Arrange 2x2
        top = np.concatenate([panels[0], panels[1]], axis=1)
        bottom = np.concatenate([panels[2], panels[3]], axis=1)
        grid = np.concatenate([top, bottom], axis=0)

        # BGR for OpenCV
        grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
        writer.write(grid_bgr)

        if (render_idx + 1) % 10 == 0 or render_idx == total - 1:
            print(f"  Frame {render_idx + 1}/{total}")

    writer.release()
    print(f"\nSaved: {out_path}")
    print(f"  Duration: {total / args.fps:.1f}s at {args.fps}fps")


if __name__ == "__main__":
    main()
