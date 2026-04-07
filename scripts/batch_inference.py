"""Batch SAM 3D Body inference on downloaded pitch clips.

Loads the model once, iterates through clips for a game,
runs inference on each, and stores mesh/skeleton data in the pitch database.

Usage:
    # Process all clips for a game
    python scripts/batch_inference.py --game-pk 813024

    # Process a specific pitch
    python scripts/batch_inference.py --play-id 02ec65f0-9054-3c9c-a72a-aaa3c610f0c9

    # Skip already-processed pitches (default: true)
    python scripts/batch_inference.py --game-pk 813024 --reprocess

    # Use skeleton mode (faster, no mesh rendering)
    python scripts/batch_inference.py --game-pk 813024 --mode skeleton
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

# macOS uses CGL, not EGL
os.environ.pop("PYOPENGL_PLATFORM", None)

import cv2
import numpy as np

sys.path.insert(0, "/tmp/sam-3d-body")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.data.pitch_db import PitchDB, MeshData


# ---------------------------------------------------------------------------
# Detection continuity (shared by both backends)
# ---------------------------------------------------------------------------
#
# Original behavior was "pick the largest detection per frame", which on
# baseball clips can flip between pitcher / catcher / batter / umpire as
# the silhouettes change during the windup. That flip causes the model to
# regress a body in a different orientation for a few frames, then snap
# back when the largest detection becomes the pitcher again — the exact
# "good → wrong → good" jank we see in Blender.
#
# _select_bbox locks onto whichever person we picked on the first valid
# frame, then on every subsequent frame chooses the candidate whose center
# is closest to the previous bbox's center. If all candidates jump too far
# (more than max_jump_frac * frame_width away), we keep the previous bbox
# rather than swapping people.

def _resolve_clip_path(db_path: str) -> tuple[str, bool]:
    """Prefer a sibling `_trimmed.mp4` if one exists.

    PitchDB stores the raw video path from the fetch manifest. The trim
    pipeline (fetch_savant_clips.py:_trim_clip) writes `foo_trimmed.mp4`
    alongside `foo.mp4`. Inference should always use the trimmed file when
    it's available — fewer frames, no broadcast-cut garbage.

    Returns (resolved_path, used_trimmed).
    """
    raw = Path(db_path)
    trimmed = raw.with_name(raw.stem + "_trimmed.mp4")
    if trimmed.exists() and trimmed.stat().st_size > 10_000:
        return str(trimmed), True
    return str(raw), False


def _bbox_center(box):
    return (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))


def _select_bbox(
    detections,
    prev_bbox,
    frame_w: int,
    frame_h: int,
    max_jump_frac: float = 0.30,
):
    """Pick the detection box that continues the same subject across frames.

    detections: list of [x1, y1, x2, y2] candidate boxes (any order).
    prev_bbox:  the box used on the previous frame, or None on the first frame.
    Returns:    a single [x1, y1, x2, y2] box (or prev_bbox if all candidates
                are too far away). On the first frame with no detections, falls
                back to the full image so the model still has something to chew
                on instead of crashing.
    """
    if not detections:
        if prev_bbox is not None:
            return prev_bbox
        return [0, 0, frame_w, frame_h]

    if prev_bbox is None:
        # First frame: pick the largest box (matches the old default).
        return max(
            detections,
            key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
        )

    px, py = _bbox_center(prev_bbox)
    max_jump_sq = (max_jump_frac * frame_w) ** 2

    best = None
    best_d2 = float("inf")
    for box in detections:
        cx, cy = _bbox_center(box)
        d2 = (cx - px) ** 2 + (cy - py) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best = box

    if best is None or best_d2 > max_jump_sq:
        return prev_bbox  # Stay locked rather than jumping to a different person.
    return best


# ---------------------------------------------------------------------------
# Backend: PyTorch / MPS
# ---------------------------------------------------------------------------

def load_model_pytorch(
    checkpoint_path: str = "/tmp/sam3d-weights/model.ckpt",
    mhr_path: str = "/tmp/sam3d-weights/assets/mhr_model.pt",
    device: str = "mps",
):
    """Load PyTorch SAM 3D Body + person detector."""
    import torch
    import torchvision
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator

    dev = torch.device(device)
    print(f"Loading SAM 3D Body (PyTorch) on {dev}...")
    t0 = time.time()

    model, cfg = load_sam_3d_body(checkpoint_path, device=dev, mhr_path=mhr_path)
    estimator = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)

    detector = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    ).to(dev).eval()

    print(f"Model loaded in {time.time() - t0:.1f}s")
    return estimator, detector, dev


def process_clip_pytorch(
    video_path: str,
    estimator,
    detector,
    device,
    det_threshold: float = 0.5,
) -> tuple[MeshData | None, float]:
    """Run PyTorch/MPS inference on a single clip."""
    import torch

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Cannot open {video_path}")
        return None, 0.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or None

    all_vertices = []
    all_joints = []
    all_pose = []
    all_cam_t = []
    shape_params = None
    focal_length = None
    total_ms = 0.0
    prev_bbox = None  # detection-continuity tracker (see _select_bbox)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img_tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().to(device) / 255.0
        with torch.no_grad():
            preds = detector([img_tensor])[0]

        person_mask = preds["labels"] == 1
        scores = preds["scores"][person_mask]
        boxes = preds["boxes"][person_mask]
        keep = scores > det_threshold
        candidates = boxes[keep].cpu().numpy().tolist()

        chosen = _select_bbox(candidates, prev_bbox, width, height)
        prev_bbox = chosen
        bbox = np.array([chosen], dtype=np.float32)

        t0 = time.perf_counter()
        outputs = estimator.process_one_image(rgb, bboxes=bbox, inference_type="body")
        elapsed = (time.perf_counter() - t0) * 1000
        total_ms += elapsed

        if outputs:
            person = outputs[0]
            all_vertices.append(person["pred_vertices"])
            all_cam_t.append(person["pred_cam_t"])

            if "pred_keypoints_3d" in person:
                all_joints.append(person["pred_keypoints_3d"])
            else:
                all_joints.append(person["pred_keypoints_2d"])

            if "pred_pose" in person:
                all_pose.append(person["pred_pose"])

            if shape_params is None and "pred_shape" in person:
                shape_params = person["pred_shape"]
            if focal_length is None:
                focal_length = person.get("focal_length", 5000.0)

    cap.release()
    return _build_mesh_data(all_vertices, all_joints, all_pose, all_cam_t,
                            shape_params, focal_length, total_ms,
                            source_fps=source_fps)


# ---------------------------------------------------------------------------
# Backend: MLX (Apple Silicon native)
# ---------------------------------------------------------------------------

def load_model_mlx(weights_dir: str = "/tmp/sam3d-mlx-weights/"):
    """Load MLX SAM 3D Body estimator."""
    from sam3d_mlx.estimator import SAM3DBodyEstimator

    print(f"Loading SAM 3D Body (MLX) from {weights_dir}...")
    t0 = time.time()
    estimator = SAM3DBodyEstimator(weights_dir)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Warm up person detector
    from sam3d_mlx.estimator import detect_persons_cached
    detect_persons_cached(np.zeros((100, 100, 3), dtype=np.uint8))

    return estimator


def process_clip_mlx(
    video_path: str,
    estimator,
    det_threshold: float = 0.5,
) -> tuple[MeshData | None, float]:
    """Run MLX inference on a single clip."""
    from sam3d_mlx.estimator import detect_persons_cached

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Cannot open {video_path}")
        return None, 0.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or None

    all_vertices = []
    all_joints = []
    all_pose = []
    all_cam_t = []
    shape_params = None
    total_ms = 0.0
    prev_bbox = None  # detection-continuity tracker (see _select_bbox)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect persons, then lock onto the same one across frames.
        detections = detect_persons_cached(rgb, threshold=det_threshold)
        bbox = _select_bbox(detections, prev_bbox, width, height)
        prev_bbox = bbox

        t0 = time.perf_counter()
        result = estimator.predict(rgb, bbox, auto_detect=False)
        elapsed = (time.perf_counter() - t0) * 1000
        total_ms += elapsed

        all_vertices.append(result["pred_vertices"])
        all_joints.append(result["pred_keypoints_3d"])
        all_cam_t.append(result["pred_camera"])
        all_pose.append(result["pred_pose"])

        if shape_params is None:
            shape_params = result["pred_shape"]

    cap.release()

    # Focal length: MLX uses image diagonal as default
    focal_length = math.sqrt(height**2 + width**2)

    return _build_mesh_data(all_vertices, all_joints, all_pose, all_cam_t,
                            shape_params, focal_length, total_ms,
                            source_fps=source_fps)


# ---------------------------------------------------------------------------
# Shared MeshData builder
# ---------------------------------------------------------------------------

def _build_mesh_data(
    all_vertices, all_joints, all_pose, all_cam_t,
    shape_params, focal_length, total_ms,
    source_fps: float | None = None,
) -> tuple[MeshData | None, float]:
    """Build MeshData from per-frame arrays collected by either backend."""
    if not all_vertices:
        return None, total_ms

    vertices = np.stack(all_vertices)
    cam_t = np.stack(all_cam_t)

    joints = np.stack(all_joints) if all_joints else np.zeros((len(all_vertices), 70, 3))
    if joints.shape[-1] == 2:
        joints = np.concatenate([joints, np.zeros((*joints.shape[:-1], 1))], axis=-1)

    if all_pose:
        pose_params = np.stack(all_pose)
    else:
        pose_params = np.zeros((len(all_vertices), 136))

    if shape_params is None:
        shape_params = np.zeros(45)

    mesh = MeshData(
        vertices=vertices,
        joints_mhr70=joints[:, :70, :] if joints.shape[1] >= 70 else joints,
        pose_params=pose_params,
        shape_params=shape_params,
        cam_t=cam_t,
        focal_length=float(focal_length) if focal_length is not None else 5000.0,
        source_fps=source_fps,
    )
    return mesh, total_ms


def main():
    parser = argparse.ArgumentParser(description="Batch SAM 3D Body inference")
    parser.add_argument("--game-pk", type=int, default=None, help="Process all clips for this game")
    parser.add_argument("--play-id", type=str, default=None, help="Process a single pitch")
    parser.add_argument("--reprocess", action="store_true", help="Re-run inference even if mesh exists")
    parser.add_argument("--db", type=str, default="data/pitches.db")
    parser.add_argument("--mesh-dir", type=str, default="data/meshes")
    parser.add_argument("--backend", choices=["pytorch", "mlx"], default="mlx",
                        help="Inference backend (default: mlx)")
    parser.add_argument("--device", type=str, default="mps",
                        help="PyTorch device (ignored for MLX)")
    parser.add_argument("--weights", type=str, default=None,
                        help="Weights path (default depends on backend)")
    parser.add_argument("--det-threshold", type=float, default=0.5)
    args = parser.parse_args()

    db = PitchDB(args.db, args.mesh_dir)

    # Get pitches to process
    if args.play_id:
        pitch = db.get_pitch(args.play_id)
        if not pitch:
            print(f"No pitch found: {args.play_id}")
            sys.exit(1)
        pitches = [pitch]
    elif args.game_pk:
        pitches = db.get_game_pitches(args.game_pk)
    else:
        parser.error("Provide --game-pk or --play-id")

    # Filter to pitches with video and optionally without mesh
    if not args.reprocess:
        pitches = [p for p in pitches if p["video_path"] and not p["mesh_path"]]
    else:
        pitches = [p for p in pitches if p["video_path"]]

    if not pitches:
        print("No pitches to process.")
        sys.exit(0)

    print(f"Backend: {args.backend}")
    print(f"Pitches to process: {len(pitches)}")

    est_frames = len(pitches) * 370
    est_ms_per_frame = 670 if args.backend == "pytorch" else 900
    est_minutes = (est_frames * est_ms_per_frame / 1000) / 60
    print(f"Estimated time: ~{est_minutes:.0f} minutes ({est_frames} frames at ~{est_ms_per_frame}ms/frame)")
    print()

    # Load model once
    if args.backend == "mlx":
        weights = args.weights or "/tmp/sam3d-mlx-weights/"
        estimator = load_model_mlx(weights)
        detector = device = None
    else:
        estimator, detector, device = load_model_pytorch(device=args.device)
    print()

    total_start = time.time()
    processed = 0
    failed = 0

    for i, pitch in enumerate(pitches):
        play_id = pitch["play_id"]
        raw_vpath = pitch["video_path"]
        ptype = pitch["pitch_type"] or "?"
        inn = pitch["inning"]
        batter = pitch["batter_name"] or "?"

        print(f"[{i+1}/{len(pitches)}] Inn {inn} | {ptype} → {batter} ({play_id[:8]})")

        vpath, used_trimmed = _resolve_clip_path(raw_vpath)
        if used_trimmed:
            print(f"  using trimmed clip: {Path(vpath).name}")

        if not Path(vpath).exists():
            print(f"  Video missing: {vpath}")
            failed += 1
            continue

        if args.backend == "mlx":
            mesh, inference_ms = process_clip_mlx(
                vpath, estimator, args.det_threshold
            )
        else:
            mesh, inference_ms = process_clip_pytorch(
                vpath, estimator, detector, device, args.det_threshold
            )

        if mesh is None:
            print(f"  No body detected")
            failed += 1
            continue

        db.update_mesh(play_id, mesh, inference_time_ms=inference_ms)
        processed += 1

        fps = mesh.vertices.shape[0] / (inference_ms / 1000) if inference_ms > 0 else 0
        print(f"  {mesh.vertices.shape[0]} frames, {inference_ms/1000:.1f}s ({fps:.1f} fps)")

    elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"Batch complete ({args.backend})")
    print(f"{'='*60}")
    print(f"  Processed: {processed}/{len(pitches)}")
    print(f"  Failed:    {failed}")
    print(f"  Total:     {elapsed/60:.1f} minutes")

    s = db.summary()
    print(f"  DB total:  {s['total_pitches']} pitches, {s['with_mesh']} with mesh")

    db.close()


if __name__ == "__main__":
    main()
