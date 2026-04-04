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
import os
import sys
import time
from pathlib import Path

# macOS uses CGL, not EGL
os.environ.pop("PYOPENGL_PLATFORM", None)

import cv2
import numpy as np
import torch

sys.path.insert(0, "/tmp/sam-3d-body")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from backend.app.data.pitch_db import PitchDB, MeshData


def load_model(
    checkpoint_path: str = "/tmp/sam3d-weights/model.ckpt",
    mhr_path: str = "/tmp/sam3d-weights/assets/mhr_model.pt",
    device: str = "mps",
):
    """Load SAM 3D Body model and person detector. Returns (estimator, detector, device)."""
    import torchvision

    dev = torch.device(device)
    print(f"Loading SAM 3D Body on {dev}...")
    t0 = time.time()

    model, cfg = load_sam_3d_body(checkpoint_path, device=dev, mhr_path=mhr_path)
    estimator = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)

    detector = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    ).to(dev).eval()

    print(f"Model loaded in {time.time() - t0:.1f}s")
    return estimator, detector, dev


def process_clip(
    video_path: str,
    estimator: SAM3DBodyEstimator,
    detector,
    device: torch.device,
    det_threshold: float = 0.5,
) -> tuple[MeshData | None, float]:
    """Run inference on a single clip.

    Returns (MeshData or None, total_inference_ms).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Cannot open {video_path}")
        return None, 0.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    all_vertices = []
    all_joints = []
    all_pose = []
    all_cam_t = []
    shape_params = None
    focal_length = None
    total_ms = 0.0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect person
        img_tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().to(device) / 255.0
        with torch.no_grad():
            preds = detector([img_tensor])[0]

        person_mask = preds["labels"] == 1
        scores = preds["scores"][person_mask]
        boxes = preds["boxes"][person_mask]
        keep = scores > det_threshold
        boxes_np = boxes[keep].cpu().numpy()

        if len(boxes_np) == 0:
            boxes_np = np.array([[0, 0, width, height]])

        # Pick largest person (likely the pitcher)
        areas = (boxes_np[:, 2] - boxes_np[:, 0]) * (boxes_np[:, 3] - boxes_np[:, 1])
        best = np.argmax(areas)
        bbox = boxes_np[best:best+1]

        t0 = time.perf_counter()
        outputs = estimator.process_one_image(rgb, bboxes=bbox, inference_type="body")
        elapsed = (time.perf_counter() - t0) * 1000
        total_ms += elapsed

        if outputs:
            person = outputs[0]
            all_vertices.append(person["pred_vertices"])
            all_cam_t.append(person["pred_cam_t"])

            kp = person["pred_keypoints_2d"]
            # MHR70 joints are in 3D from the model
            if "pred_keypoints_3d" in person:
                all_joints.append(person["pred_keypoints_3d"])
            else:
                all_joints.append(kp)

            if "pred_pose" in person:
                all_pose.append(person["pred_pose"])

            if shape_params is None and "pred_shape" in person:
                shape_params = person["pred_shape"]
            if focal_length is None:
                focal_length = person.get("focal_length", 5000.0)

        frame_idx += 1

    cap.release()

    if not all_vertices:
        return None, total_ms

    vertices = np.stack(all_vertices)  # (T, N, 3)
    cam_t = np.stack(all_cam_t)        # (T, 3)

    # Joints: use 3D keypoints, fall back to 2D
    joints = np.stack(all_joints) if all_joints else np.zeros((len(all_vertices), 70, 3))
    if joints.shape[-1] == 2:
        # Pad 2D to 3D with zeros
        joints = np.concatenate([joints, np.zeros((*joints.shape[:-1], 1))], axis=-1)

    # Pose params
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
    )
    return mesh, total_ms


def main():
    parser = argparse.ArgumentParser(description="Batch SAM 3D Body inference")
    parser.add_argument("--game-pk", type=int, default=None, help="Process all clips for this game")
    parser.add_argument("--play-id", type=str, default=None, help="Process a single pitch")
    parser.add_argument("--reprocess", action="store_true", help="Re-run inference even if mesh exists")
    parser.add_argument("--db", type=str, default="data/pitches.db")
    parser.add_argument("--mesh-dir", type=str, default="data/meshes")
    parser.add_argument("--device", type=str, default="mps")
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

    print(f"Pitches to process: {len(pitches)}")

    # Estimate time
    est_frames = len(pitches) * 370  # ~370 frames per 6s clip at 60fps
    est_minutes = (est_frames * 0.67) / 60  # 670ms/frame
    print(f"Estimated time: ~{est_minutes:.0f} minutes ({est_frames} frames at ~670ms/frame)")
    print()

    # Load model once
    estimator, detector, device = load_model(device=args.device)
    print()

    total_start = time.time()
    processed = 0
    failed = 0

    for i, pitch in enumerate(pitches):
        play_id = pitch["play_id"]
        vpath = pitch["video_path"]
        ptype = pitch["pitch_type"] or "?"
        inn = pitch["inning"]
        batter = pitch["batter_name"] or "?"

        print(f"[{i+1}/{len(pitches)}] Inn {inn} | {ptype} → {batter} ({play_id[:8]})")

        if not Path(vpath).exists():
            print(f"  Video missing: {vpath}")
            failed += 1
            continue

        mesh, inference_ms = process_clip(
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
    print(f"Batch complete")
    print(f"{'='*60}")
    print(f"  Processed: {processed}/{len(pitches)}")
    print(f"  Failed:    {failed}")
    print(f"  Total:     {elapsed/60:.1f} minutes")

    s = db.summary()
    print(f"  DB total:  {s['total_pitches']} pitches, {s['with_mesh']} with mesh")

    db.close()


if __name__ == "__main__":
    main()
