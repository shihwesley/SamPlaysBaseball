"""Test the fixed MLX body model on a video clip.

Produces two outputs:
1. Skeleton overlay video
2. 3D mesh overlay video (projected wireframe on original frame)
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/Users/quartershots/Source/SamPlaysBaseball")

CLIP = "data/clips/813024/inn3_ab23_p1_SL_5b1e2524.mp4"
OUT_DIR = Path("data/output/813024/test_run")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SECONDS = 4
WEIGHTS = "/tmp/sam3d-mlx-weights/"

# --- Video info ---
cap = cv2.VideoCapture(CLIP)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
max_frames = int(fps * MAX_SECONDS)
print(f"Clip: {CLIP}")
print(f"  {width}x{height} @ {fps:.1f}fps, processing {max_frames} frames ({MAX_SECONDS}s)")

# --- Load model ---
print(f"\nLoading MLX model...")
t0 = time.time()
from sam3d_mlx.estimator import SAM3DBodyEstimator
estimator = SAM3DBodyEstimator(WEIGHTS)
print(f"Model loaded in {time.time()-t0:.1f}s")

# Get mesh faces from the model
import mlx.core as mx
faces = np.array(estimator.model.head_pose.body_model.base_shape)  # wrong
# Actually get faces from the JIT model or safetensors
from safetensors.numpy import safe_open
with safe_open(f"{WEIGHTS}/model.safetensors", framework="numpy") as f:
    for key in f.keys():
        if "faces" in key.lower() and "texcoord" not in key.lower():
            faces = f.get_tensor(key)
            print(f"Loaded faces from {key}: {faces.shape}")
            break

# --- Person detection ---
print("\nLoading person detector...")
t0 = time.time()
from sam3d_mlx.estimator import detect_persons_cached
dummy = np.zeros((100, 100, 3), dtype=np.uint8)
detect_persons_cached(dummy)
print(f"Detector loaded in {time.time()-t0:.1f}s")

# Baseball mound region hint (approximate center of frame)
target_region = [width * 0.3, height * 0.1, width * 0.7, height * 0.9]

# --- Projection helpers ---
from sam3d_mlx.video import (
    project_keypoints_perspective, draw_skeleton, draw_bbox,
    track_person, SKELETON_PAIRS
)


# --- Process frames ---
print(f"\nProcessing {max_frames} frames...")

# Output writers
skel_path = str(OUT_DIR / "skeleton.mp4")
mesh_path = str(OUT_DIR / "mesh.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
skel_writer = cv2.VideoWriter(skel_path, fourcc, fps, (width, height))
mesh_writer = cv2.VideoWriter(mesh_path, fourcc, fps, (width, height))

import math
import trimesh
import pyrender

# Match existing mesh render style: light blue-grey, 3-point lighting
LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def render_mesh_pyrender(vertices, cam_t, focal_length, frame_bgr, faces):
    """Render mesh with pyrender (matches ab_test_mesh.py style)."""
    h, w = frame_bgr.shape[:2]
    image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    camera_translation = cam_t.copy()
    camera_translation[0] *= -1.0

    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0, alphaMode="OPAQUE",
        baseColorFactor=(LIGHT_BLUE[2], LIGHT_BLUE[1], LIGHT_BLUE[0], 1.0))
    mesh = trimesh.Trimesh(vertices.copy(), faces.copy())
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0]))
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh)
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = camera_translation
    scene.add(pyrender.IntrinsicsCamera(
        fx=focal_length, fy=focal_length, cx=w / 2, cy=h / 2, zfar=1e12),
        pose=camera_pose)

    for theta, phi in zip([np.pi / 6] * 3, [0, 2 * np.pi / 3, 4 * np.pi / 3]):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)
        z = np.array([xp, yp, zp])
        z /= np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1, 0, 0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        mat = np.eye(4)
        mat[:3, :3] = np.c_[x, y, z]
        scene.add_node(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=mat))

    renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
    color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()

    valid = (depth > 0).astype(np.float32)[:, :, None]
    color_f = color[:, :, :3].astype(np.float32) / 255.0
    output = color_f * valid + image * (1 - valid)
    return (output * 255).astype(np.uint8)


def compute_cam_t(camera, bbox, img_w, img_h, fov_deg=60.0):
    """Convert (scale, tx, ty) + bbox to camera translation."""
    pred_cam = camera.copy()
    pred_cam[[0, 2]] *= -1
    s, tx, ty = pred_cam
    bbox_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    focal_length = img_h / (2 * math.tan(math.radians(fov_deg / 2)))
    bbox_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
    bs = bbox_size * s + 1e-8
    tz = 2 * focal_length / bs
    cx = 2 * (bbox_center[0] - img_w / 2) / bs
    cy = 2 * (bbox_center[1] - img_h / 2) / bs
    return np.array([tx + cx, ty + cy, tz]), focal_length


tracked_bbox = target_region
frame_times = []
frame_idx = 0

while frame_idx < max_frames:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect person
    detections = detect_persons_cached(rgb)
    frame_bbox = track_person(detections, tracked_bbox)
    if frame_bbox is not None:
        tracked_bbox = frame_bbox
    else:
        frame_bbox = [0, 0, width, height]

    # Inference
    t0 = time.perf_counter()
    result = estimator.predict(rgb, frame_bbox, auto_detect=False)
    dt = time.perf_counter() - t0
    frame_times.append(dt)

    kp_3d = result["pred_keypoints_3d"]
    camera = result["pred_camera"]
    verts_3d = result["pred_vertices"]
    used_bbox = result.get("bbox", frame_bbox)

    # Project keypoints to 2D
    kp_2d = project_keypoints_perspective(kp_3d, camera, used_bbox, width, height)

    cam_t, focal = compute_cam_t(camera, used_bbox, width, height)
    ms = dt * 1000

    # --- Output 1: Skeleton ---
    skel_frame = frame.copy()
    skel_frame = draw_bbox(skel_frame, used_bbox)
    skel_frame = draw_skeleton(skel_frame, kp_2d)
    cv2.putText(skel_frame, f"MLX: {ms:.0f}ms  Frame {frame_idx}/{max_frames}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    skel_writer.write(skel_frame)

    # --- Output 2: Mesh (pyrender, grey translucent) ---
    mesh_rgb = render_mesh_pyrender(verts_3d, cam_t, focal, frame, faces)
    mesh_frame = cv2.cvtColor(mesh_rgb, cv2.COLOR_RGB2BGR)
    cv2.putText(mesh_frame, f"MLX Mesh: {ms:.0f}ms  Frame {frame_idx}/{max_frames}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    mesh_writer.write(mesh_frame)

    # Progress
    if frame_idx % 20 == 0 or frame_idx == 0:
        avg_ms = np.mean(frame_times[-20:]) * 1000
        eta = (max_frames - frame_idx) * avg_ms / 1000
        print(f"  Frame {frame_idx:3d}/{max_frames}  {avg_ms:.0f}ms/frame  ETA: {eta:.0f}s")

    frame_idx += 1

cap.release()
skel_writer.release()
mesh_writer.release()

# Summary
times = np.array(frame_times)
print(f"\nDone!")
print(f"  Frames: {frame_idx}")
print(f"  Median: {np.median(times)*1000:.0f}ms/frame")
print(f"  Total:  {np.sum(times):.1f}s inference")
print(f"\nOutputs:")
print(f"  Skeleton: {skel_path} ({Path(skel_path).stat().st_size/1024**2:.1f}MB)")
print(f"  Mesh:     {mesh_path} ({Path(mesh_path).stat().st_size/1024**2:.1f}MB)")
