"""Benchmark: PyTorch/MPS vs MLX on the same clip.

Measures pure inference time (no rendering, no detection overhead).
Both pipelines use the same frames, same bboxes, same warmup protocol.
"""

import gc
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/Users/quartershots/Source/SamPlaysBaseball")
sys.path.insert(0, "/tmp/sam-3d-body")

CLIP = "data/clips/813024/inn3_ab23_p1_SL_5b1e2524.mp4"
MAX_SECONDS = 4
WARMUP_FRAMES = 5  # discard first N frames from timing

# --- Extract frames + detect bboxes once ---
print("=" * 60)
print("BENCHMARK: PyTorch/MPS vs MLX")
print("=" * 60)

cap = cv2.VideoCapture(CLIP)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
max_frames = int(fps * MAX_SECONDS)
print(f"Clip: {CLIP}")
print(f"  {width}x{height} @ {fps:.1f}fps, {max_frames} frames ({MAX_SECONDS}s)")

# Read all frames into memory
frames_rgb = []
frame_idx = 0
while frame_idx < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame_idx += 1
cap.release()
print(f"  Loaded {len(frames_rgb)} frames into memory")

# Detect bboxes once using torchvision (shared between both)
print("\nDetecting persons (shared between both pipelines)...")
import torch
import torchvision

device = torch.device("mps")
detector = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
    weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
).to(device).eval()

bboxes = []
for i, rgb in enumerate(frames_rgb):
    img_t = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().to(device) / 255.0
    with torch.no_grad():
        preds = detector([img_t])[0]
    person_mask = preds["labels"] == 1
    scores = preds["scores"][person_mask]
    boxes = preds["boxes"][person_mask]
    keep = scores > 0.5
    boxes_np = boxes[keep].cpu().numpy()
    if len(boxes_np) == 0:
        bboxes.append([0, 0, width, height])
    else:
        areas = (boxes_np[:, 2] - boxes_np[:, 0]) * (boxes_np[:, 3] - boxes_np[:, 1])
        best = np.argmax(areas)
        bboxes.append(boxes_np[best].tolist())

del detector
torch.mps.empty_cache()
gc.collect()
print(f"  Detected {len(bboxes)} bboxes")

# ============================================================
# PYTORCH / MPS
# ============================================================
print("\n" + "=" * 60)
print("PYTORCH / MPS")
print("=" * 60)

print("Loading model...")
t0 = time.time()
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator

model, cfg = load_sam_3d_body(
    "/tmp/sam3d-weights/model.ckpt",
    device=torch.device("mps"),
    mhr_path="/tmp/sam3d-weights/assets/mhr_model.pt",
)
estimator_pt = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)
load_time_pt = time.time() - t0
print(f"  Loaded in {load_time_pt:.1f}s")

# Warmup
print(f"  Warming up ({WARMUP_FRAMES} frames)...")
for i in range(WARMUP_FRAMES):
    bbox_np = np.array([bboxes[i]], dtype=np.float32)
    _ = estimator_pt.process_one_image(frames_rgb[i], bboxes=bbox_np, inference_type="body")
torch.mps.synchronize()

# Benchmark
print(f"  Benchmarking {len(frames_rgb) - WARMUP_FRAMES} frames...")
pt_times = []
for i in range(WARMUP_FRAMES, len(frames_rgb)):
    bbox_np = np.array([bboxes[i]], dtype=np.float32)
    torch.mps.synchronize()
    t0 = time.perf_counter()
    outputs = estimator_pt.process_one_image(frames_rgb[i], bboxes=bbox_np, inference_type="body")
    torch.mps.synchronize()
    dt = time.perf_counter() - t0
    pt_times.append(dt)

    if (i - WARMUP_FRAMES) % 40 == 0:
        print(f"    Frame {i}: {dt*1000:.0f}ms")

pt_times = np.array(pt_times)
print(f"\n  Results:")
print(f"    Median:  {np.median(pt_times)*1000:.1f}ms")
print(f"    Mean:    {np.mean(pt_times)*1000:.1f}ms")
print(f"    P5:      {np.percentile(pt_times, 5)*1000:.1f}ms")
print(f"    P95:     {np.percentile(pt_times, 95)*1000:.1f}ms")
print(f"    Total:   {np.sum(pt_times):.1f}s")
print(f"    FPS:     {len(pt_times)/np.sum(pt_times):.1f}")

# Save a reference frame for visual comparison
pt_last_output = outputs[0] if outputs else None
if pt_last_output:
    np.save("data/output/813024/test_run/pt_verts_bench.npy", pt_last_output["pred_vertices"])

# --- PyTorch inference + rendering ---
print(f"\n  Benchmarking inference + mesh render ({len(frames_rgb) - WARMUP_FRAMES} frames)...")

import os
os.environ.pop("PYOPENGL_PLATFORM", None)
import pyrender
import trimesh

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
pt_faces = estimator_pt.faces

def render_mesh_pt(vertices, cam_t, focal_length, frame_bgr, faces):
    h, w = frame_bgr.shape[:2]
    image = frame_bgr.astype(np.float32) / 255.0
    ct = cam_t.copy(); ct[0] *= -1.0
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0, alphaMode="OPAQUE",
        baseColorFactor=(LIGHT_BLUE[2], LIGHT_BLUE[1], LIGHT_BLUE[0], 1.0))
    mesh = trimesh.Trimesh(vertices.copy(), faces.copy())
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0]))
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
    scene = pyrender.Scene(bg_color=[0,0,0,0], ambient_light=(0.3,0.3,0.3))
    scene.add(mesh)
    cp = np.eye(4); cp[:3, 3] = ct
    scene.add(pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length, cx=w/2, cy=h/2, zfar=1e12), pose=cp)
    for theta, phi in zip([np.pi/6]*3, [0, 2*np.pi/3, 4*np.pi/3]):
        xp, yp, zp = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
        z = np.array([xp, yp, zp]); z /= np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0: x = np.array([1, 0, 0])
        x /= np.linalg.norm(x); y = np.cross(z, x)
        mat = np.eye(4); mat[:3, :3] = np.c_[x, y, z]
        scene.add_node(pyrender.Node(light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0), matrix=mat))
    renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
    color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()
    valid = (depth > 0).astype(np.float32)[:, :, None]
    color_f = color[:, :, :3].astype(np.float32) / 255.0
    output = color_f * valid + image * (1 - valid)
    return (output * 255).astype(np.uint8)

pt_render_times = []
for i in range(WARMUP_FRAMES, len(frames_rgb)):
    bbox_np = np.array([bboxes[i]], dtype=np.float32)
    torch.mps.synchronize()
    t0 = time.perf_counter()
    outputs = estimator_pt.process_one_image(frames_rgb[i], bboxes=bbox_np, inference_type="body")
    torch.mps.synchronize()
    if outputs:
        p = outputs[0]
        _ = render_mesh_pt(p["pred_vertices"], p["pred_cam_t"], p["focal_length"],
                           cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2BGR), pt_faces)
    dt = time.perf_counter() - t0
    pt_render_times.append(dt)

    if (i - WARMUP_FRAMES) % 40 == 0:
        print(f"    Frame {i}: {dt*1000:.0f}ms")

pt_render_times = np.array(pt_render_times)
print(f"  Inference + render median: {np.median(pt_render_times)*1000:.1f}ms")

# Free PyTorch memory
del estimator_pt, model
torch.mps.empty_cache()
gc.collect()
time.sleep(2)  # let GPU cool

# ============================================================
# MLX
# ============================================================
print("\n" + "=" * 60)
print("MLX")
print("=" * 60)

print("Loading model...")
t0 = time.time()
from sam3d_mlx.estimator import SAM3DBodyEstimator as MLXEstimator
import mlx.core as mx

estimator_mlx = MLXEstimator("/tmp/sam3d-mlx-weights/")
load_time_mlx = time.time() - t0
print(f"  Loaded in {load_time_mlx:.1f}s")

# Warmup
print(f"  Warming up ({WARMUP_FRAMES} frames)...")
for i in range(WARMUP_FRAMES):
    _ = estimator_mlx.predict(frames_rgb[i], bboxes[i], auto_detect=False)
mx.eval(mx.array([0]))  # sync

# Benchmark
print(f"  Benchmarking {len(frames_rgb) - WARMUP_FRAMES} frames...")
mlx_times = []
for i in range(WARMUP_FRAMES, len(frames_rgb)):
    mx.eval(mx.array([0]))  # sync before timing
    t0 = time.perf_counter()
    result = estimator_mlx.predict(frames_rgb[i], bboxes[i], auto_detect=False)
    mx.eval(mx.array([0]))  # sync after
    dt = time.perf_counter() - t0
    mlx_times.append(dt)

    if (i - WARMUP_FRAMES) % 40 == 0:
        print(f"    Frame {i}: {dt*1000:.0f}ms")

mlx_times = np.array(mlx_times)
print(f"\n  Results:")
print(f"    Median:  {np.median(mlx_times)*1000:.1f}ms")
print(f"    Mean:    {np.mean(mlx_times)*1000:.1f}ms")
print(f"    P5:      {np.percentile(mlx_times, 5)*1000:.1f}ms")
print(f"    P95:     {np.percentile(mlx_times, 95)*1000:.1f}ms")
print(f"    Total:   {np.sum(mlx_times):.1f}s")
print(f"    FPS:     {len(mlx_times)/np.sum(mlx_times):.1f}")

# Save reference frame
mlx_verts = result["pred_vertices"]
np.save("data/output/813024/test_run/mlx_verts_bench.npy", mlx_verts)

# --- MLX inference + rendering ---
print(f"\n  Benchmarking inference + mesh render ({len(frames_rgb) - WARMUP_FRAMES} frames)...")
import math
from safetensors.numpy import safe_open
with safe_open("/tmp/sam3d-mlx-weights/model.safetensors", framework="numpy") as f:
    for key in f.keys():
        if "faces" in key.lower() and "texcoord" not in key.lower():
            mlx_faces = f.get_tensor(key)
            break

def compute_cam_t_mlx(camera, bbox, img_w, img_h):
    pred_cam = camera.copy(); pred_cam[[0, 2]] *= -1
    s, tx, ty = pred_cam
    focal = math.sqrt(img_h**2 + img_w**2)
    bbox_center = np.array([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2])
    bbox_size = max(bbox[2]-bbox[0], bbox[3]-bbox[1])
    bs = bbox_size * s + 1e-8
    tz = 2 * focal / bs
    cx = 2 * (bbox_center[0] - img_w/2) / bs
    cy = 2 * (bbox_center[1] - img_h/2) / bs
    return np.array([tx+cx, ty+cy, tz]), focal

def render_mesh_mlx(vertices, cam_t, focal, frame_bgr, faces):
    h, w = frame_bgr.shape[:2]
    image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    ct = cam_t.copy(); ct[0] *= -1.0
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0, alphaMode="OPAQUE",
        baseColorFactor=(LIGHT_BLUE[2], LIGHT_BLUE[1], LIGHT_BLUE[0], 1.0))
    mesh = trimesh.Trimesh(vertices.copy(), faces.copy())
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0]))
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
    scene = pyrender.Scene(bg_color=[0,0,0,0], ambient_light=(0.3,0.3,0.3))
    scene.add(mesh)
    cp = np.eye(4); cp[:3, 3] = ct
    scene.add(pyrender.IntrinsicsCamera(fx=focal, fy=focal, cx=w/2, cy=h/2, zfar=1e12), pose=cp)
    for theta, phi in zip([np.pi/6]*3, [0, 2*np.pi/3, 4*np.pi/3]):
        xp, yp, zp = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
        z = np.array([xp, yp, zp]); z /= np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0: x = np.array([1, 0, 0])
        x /= np.linalg.norm(x); y = np.cross(z, x)
        mat = np.eye(4); mat[:3, :3] = np.c_[x, y, z]
        scene.add_node(pyrender.Node(light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0), matrix=mat))
    renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
    color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()
    valid = (depth > 0).astype(np.float32)[:, :, None]
    color_f = color[:, :, :3].astype(np.float32) / 255.0
    output = color_f * valid + image * (1 - valid)
    return (output * 255).astype(np.uint8)

mlx_render_times = []
for i in range(WARMUP_FRAMES, len(frames_rgb)):
    mx.eval(mx.array([0]))
    t0 = time.perf_counter()
    result = estimator_mlx.predict(frames_rgb[i], bboxes[i], auto_detect=False)
    mx.eval(mx.array([0]))
    cam_t, focal = compute_cam_t_mlx(result["pred_camera"], bboxes[i], width, height)
    frame_bgr = cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2BGR)
    _ = render_mesh_mlx(result["pred_vertices"], cam_t, focal, frame_bgr, mlx_faces)
    dt = time.perf_counter() - t0
    mlx_render_times.append(dt)

    if (i - WARMUP_FRAMES) % 40 == 0:
        print(f"    Frame {i}: {dt*1000:.0f}ms")

mlx_render_times = np.array(mlx_render_times)
print(f"  Inference + render median: {np.median(mlx_render_times)*1000:.1f}ms")

# ============================================================
# COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)

pt_med = np.median(pt_times) * 1000
mlx_med = np.median(mlx_times) * 1000
speedup_inf = pt_med / mlx_med

pt_rmed = np.median(pt_render_times) * 1000
mlx_rmed = np.median(mlx_render_times) * 1000
speedup_render = pt_rmed / mlx_rmed

print(f"\n  --- Inference Only ---")
print(f"  {'Metric':<20} {'PyTorch/MPS':>15} {'MLX':>15} {'Ratio':>10}")
print(f"  {'-'*60}")
print(f"  {'Load time':<20} {load_time_pt:>14.1f}s {load_time_mlx:>14.1f}s {load_time_pt/load_time_mlx:>9.1f}x")
print(f"  {'Median (ms)':<20} {pt_med:>15.1f} {mlx_med:>15.1f} {speedup_inf:>9.2f}x")
print(f"  {'Mean (ms)':<20} {np.mean(pt_times)*1000:>15.1f} {np.mean(mlx_times)*1000:>15.1f} {np.mean(pt_times)/np.mean(mlx_times):>9.2f}x")
print(f"  {'P5 (ms)':<20} {np.percentile(pt_times,5)*1000:>15.1f} {np.percentile(mlx_times,5)*1000:>15.1f}")
print(f"  {'P95 (ms)':<20} {np.percentile(pt_times,95)*1000:>15.1f} {np.percentile(mlx_times,95)*1000:>15.1f}")
print(f"  {'FPS':<20} {len(pt_times)/np.sum(pt_times):>15.1f} {len(mlx_times)/np.sum(mlx_times):>15.1f}")

print(f"\n  --- Inference + Mesh Render ---")
print(f"  {'Metric':<20} {'PyTorch/MPS':>15} {'MLX':>15} {'Ratio':>10}")
print(f"  {'-'*60}")
print(f"  {'Median (ms)':<20} {pt_rmed:>15.1f} {mlx_rmed:>15.1f} {speedup_render:>9.2f}x")
print(f"  {'Mean (ms)':<20} {np.mean(pt_render_times)*1000:>15.1f} {np.mean(mlx_render_times)*1000:>15.1f} {np.mean(pt_render_times)/np.mean(mlx_render_times):>9.2f}x")
print(f"  {'FPS':<20} {len(pt_render_times)/np.sum(pt_render_times):>15.1f} {len(mlx_render_times)/np.sum(mlx_render_times):>15.1f}")

pt_render_overhead = pt_rmed - pt_med
mlx_render_overhead = mlx_rmed - mlx_med
print(f"\n  --- Render Overhead ---")
print(f"  {'PyTorch/MPS':<20} {pt_render_overhead:>10.1f}ms")
print(f"  {'MLX':<20} {mlx_render_overhead:>10.1f}ms")

print(f"\n  --- Summary ---")
if speedup_inf > 1:
    print(f"  Inference: MLX is {speedup_inf:.2f}x faster")
else:
    print(f"  Inference: PyTorch/MPS is {1/speedup_inf:.2f}x faster")
if speedup_render > 1:
    print(f"  Inf+Render: MLX is {speedup_render:.2f}x faster")
else:
    print(f"  Inf+Render: PyTorch/MPS is {1/speedup_render:.2f}x faster")

# Vertex comparison (last frame)
try:
    pt_v = np.load("data/output/813024/test_run/pt_verts_bench.npy")
    mlx_v = np.load("data/output/813024/test_run/mlx_verts_bench.npy")
    vdiff = np.linalg.norm(pt_v - mlx_v, axis=1)
    print(f"\n  Vertex comparison (last frame):")
    print(f"    Max L2 diff: {vdiff.max():.4f}")
    print(f"    Mean L2 diff: {vdiff.mean():.4f}")
except Exception as e:
    print(f"\n  Could not compare vertices: {e}")
