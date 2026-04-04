"""SAM 3D Body video pipeline — PyTorch/MPS on Apple Silicon.

Renders skeleton overlay, 3D mesh overlay, or both (4-panel).

Usage:
    # Skeleton only (fast)
    python scripts/run_pytorch_video.py -i video.mp4 --mode skeleton

    # Mesh overlay on video
    python scripts/run_pytorch_video.py -i video.mp4 --mode mesh

    # 4-panel: original | skeleton | mesh | side view
    python scripts/run_pytorch_video.py -i video.mp4 --mode full
"""

import argparse
import os
import sys
import time
from pathlib import Path

# macOS uses CGL, not EGL. Must set before any OpenGL import.
os.environ.pop("PYOPENGL_PLATFORM", None)

import cv2
import numpy as np
import torch

sys.path.insert(0, "/tmp/sam-3d-body")

from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info

# MHR70 skeleton pairs from official metadata
_NAME_TO_IDX = {v["name"]: k for k, v in mhr70_pose_info["keypoint_info"].items()}
SKELETON_PAIRS = []
SKELETON_COLORS = []
for entry in mhr70_pose_info["skeleton_info"].values():
    name_a, name_b = entry["link"]
    idx_a = _NAME_TO_IDX.get(name_a)
    idx_b = _NAME_TO_IDX.get(name_b)
    if idx_a is not None and idx_b is not None:
        SKELETON_PAIRS.append((idx_a, idx_b))
        SKELETON_COLORS.append(tuple(entry["color"]))

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def draw_skeleton(frame, keypoints_2d):
    """Draw MHR70 skeleton overlay on BGR frame."""
    h, w = frame.shape[:2]
    for (i, j), color in zip(SKELETON_PAIRS, SKELETON_COLORS):
        if i >= len(keypoints_2d) or j >= len(keypoints_2d):
            continue
        pt1 = (int(keypoints_2d[i, 0]), int(keypoints_2d[i, 1]))
        pt2 = (int(keypoints_2d[j, 0]), int(keypoints_2d[j, 1]))
        if not (0 <= pt1[0] < w and 0 <= pt1[1] < h):
            continue
        if not (0 <= pt2[0] < w and 0 <= pt2[1] < h):
            continue
        bgr = (color[2], color[1], color[0])
        cv2.line(frame, pt1, pt2, bgr, 2, cv2.LINE_AA)

    for i, pt in enumerate(keypoints_2d):
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(frame, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 3, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def draw_bbox(frame, bbox, color=(0, 200, 255), thickness=2):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return frame


def render_mesh(vertices, cam_t, focal_length, frame, side_view=False):
    """Render 3D mesh overlay using pyrender (macOS-compatible)."""
    import pyrender
    import trimesh

    h, w = frame.shape[:2]
    image = frame.astype(np.float32) / 255.0

    camera_translation = cam_t.copy()
    camera_translation[0] *= -1.0

    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0, alphaMode="OPAQUE",
        baseColorFactor=(LIGHT_BLUE[2], LIGHT_BLUE[1], LIGHT_BLUE[0], 1.0),
    )
    mesh = trimesh.Trimesh(vertices.copy(), _faces.copy())

    if side_view:
        rot = trimesh.transformations.rotation_matrix(np.radians(90), [0, 1, 0])
        mesh.apply_transform(rot)

    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)

    mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

    scene = pyrender.Scene(bg_color=[1, 1, 1, 0.0], ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, "mesh")

    camera_pose = np.eye(4)
    camera_pose[:3, 3] = camera_translation
    camera = pyrender.IntrinsicsCamera(
        fx=focal_length, fy=focal_length,
        cx=w / 2.0, cy=h / 2.0, zfar=1e12,
    )
    scene.add(camera, pose=camera_pose)

    # Raymond lights
    for theta, phi in zip(
        [np.pi/6]*3, [0, 2*np.pi/3, 4*np.pi/3]
    ):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)
        z = np.array([xp, yp, zp]); z /= np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0: x = np.array([1, 0, 0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        mat = np.eye(4); mat[:3, :3] = np.c_[x, y, z]
        scene.add_node(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=mat,
        ))

    renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    color = color.astype(np.float32) / 255.0
    renderer.delete()

    valid_mask = color[:, :, 3:4]
    output = color[:, :, :3] * valid_mask + (1 - valid_mask) * image

    # Convert RGB to BGR for OpenCV
    output_bgr = (output[:, :, ::-1] * 255).astype(np.uint8)
    return output_bgr


# Global reference for faces array (set during init)
_faces = None


def process_video(
    input_path: str,
    output_path: str,
    mode: str = "mesh",
    checkpoint_path: str = "/tmp/sam3d-weights/model.ckpt",
    mhr_path: str = "/tmp/sam3d-weights/assets/mhr_model.pt",
    max_frames: int = None,
    skip_frames: int = 0,
    det_threshold: float = 0.5,
):
    global _faces

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: cannot open video '{input_path}'")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if max_frames:
        total_frames = min(total_frames, max_frames)

    print(f"Input: {input_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.1f}, Frames: {total_frames}")
    print(f"  Mode: {mode}")

    # Load model on MPS
    device = torch.device("mps")
    print(f"\nLoading SAM 3D Body on {device}...")
    t0 = time.time()
    model, cfg = load_sam_3d_body(checkpoint_path, device=device, mhr_path=mhr_path)

    # Person detector
    import torchvision
    detector = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    ).to(device).eval()

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model, model_cfg=cfg,
    )
    _faces = estimator.faces
    print(f"Loaded in {time.time() - t0:.1f}s")

    # Output dimensions depend on mode
    if mode == "full":
        out_w = width * 4
    elif mode == "mesh+skeleton":
        out_w = width * 2
    else:
        out_w = width

    out_fps = fps / (skip_frames + 1) if skip_frames > 0 else fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (out_w, height))

    frame_times = []
    frame_idx = 0
    processed = 0

    print(f"\nProcessing...")

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= total_frames:
            break

        if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
            frame_idx += 1
            continue

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

        areas = (boxes_np[:, 2] - boxes_np[:, 0]) * (boxes_np[:, 3] - boxes_np[:, 1])
        best = np.argmax(areas)
        bbox = boxes_np[best:best+1]

        t0 = time.perf_counter()
        outputs = estimator.process_one_image(rgb, bboxes=bbox, inference_type="body")
        inference_time = time.perf_counter() - t0
        frame_times.append(inference_time)

        if outputs:
            person = outputs[0]
            kp_2d = person["pred_keypoints_2d"]
            if kp_2d.shape[1] > 2:
                kp_2d = kp_2d[:, :2]

            if mode == "skeleton":
                out_frame = frame.copy()
                draw_bbox(out_frame, bbox[0])
                draw_skeleton(out_frame, kp_2d)

            elif mode == "mesh":
                out_frame = render_mesh(
                    person["pred_vertices"], person["pred_cam_t"],
                    person["focal_length"], frame.copy(),
                )

            elif mode == "mesh+skeleton":
                skel_frame = frame.copy()
                draw_bbox(skel_frame, bbox[0])
                draw_skeleton(skel_frame, kp_2d)

                mesh_frame = render_mesh(
                    person["pred_vertices"], person["pred_cam_t"],
                    person["focal_length"], frame.copy(),
                )
                out_frame = np.concatenate([skel_frame, mesh_frame], axis=1)

            elif mode == "full":
                skel_frame = frame.copy()
                draw_bbox(skel_frame, bbox[0])
                draw_skeleton(skel_frame, kp_2d)

                mesh_frame = render_mesh(
                    person["pred_vertices"], person["pred_cam_t"],
                    person["focal_length"], frame.copy(),
                )

                white = np.ones_like(frame) * 255
                side_frame = render_mesh(
                    person["pred_vertices"], person["pred_cam_t"],
                    person["focal_length"], white, side_view=True,
                )

                out_frame = np.concatenate(
                    [frame, skel_frame, mesh_frame, side_frame], axis=1
                )
        else:
            if mode == "full":
                out_frame = np.concatenate([frame] * 4, axis=1)
            elif mode == "mesh+skeleton":
                out_frame = np.concatenate([frame] * 2, axis=1)
            else:
                out_frame = frame.copy()

        # Timing overlay
        fps_current = 1.0 / inference_time if inference_time > 0 else 0
        cv2.putText(
            out_frame,
            f"MPS: {inference_time*1000:.0f}ms ({fps_current:.1f} fps)",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )

        writer.write(out_frame)
        processed += 1

        if processed % 10 == 0 or processed == 1:
            avg_ms = np.mean(frame_times[-10:]) * 1000
            eta = (total_frames - frame_idx) * avg_ms / 1000
            print(f"  Frame {frame_idx:4d}/{total_frames}  {avg_ms:.0f}ms/frame  ETA: {eta:.0f}s")

        frame_idx += 1

    cap.release()
    writer.release()

    if frame_times:
        times = np.array(frame_times)
        print(f"\n{'='*60}")
        print(f"Done — {mode} mode")
        print(f"{'='*60}")
        print(f"  Frames:    {processed}")
        print(f"  Total:     {np.sum(times):.1f}s")
        print(f"  Median:    {np.median(times)*1000:.0f}ms/frame")
        print(f"  Output:    {output_path}")
        print(f"  Size:      {Path(output_path).stat().st_size / 1024**2:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM 3D Body video (PyTorch/MPS)")
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument(
        "--mode", default="mesh",
        choices=["skeleton", "mesh", "mesh+skeleton", "full"],
        help="skeleton=lines, mesh=3D body overlay, full=4-panel comparison",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--det-threshold", type=float, default=0.5)
    args = parser.parse_args()

    if args.output is None:
        stem = Path(args.input).stem
        args.output = f"output/{stem}_{args.mode}.mp4"

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    process_video(
        input_path=args.input,
        output_path=args.output,
        mode=args.mode,
        max_frames=args.max_frames,
        skip_frames=args.skip,
        det_threshold=args.det_threshold,
    )
