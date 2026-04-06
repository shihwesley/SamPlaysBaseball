"""A/B test mesh quality: current fixes vs original behavior.

Renders 10-frame mp4s for each variant:
  - mesh_A_all_fixes.mp4         (current code, all bug fixes applied)
  - mesh_B_original_cliff.mp4    (revert BUG-5: use image-dim normalization)
  - mesh_C_original_rays.mp4     (revert BUG-6: use stride sampling)
  - mesh_D_original_layernorm.mp4 (revert BUG-7: use plain LayerNorm)
  - mesh_E_all_reverted.mp4      (revert BUG-5+6+7 together)
"""

import cv2
import math
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import pyrender
import trimesh
from PIL import Image

from sam3d_mlx.estimator import SAM3DBodyEstimator, detect_persons_cached
from sam3d_mlx.video import track_person

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
CLIP = "data/clips/813024/inn1_ab5_p1_CU_02ec65f0.mp4"
TARGET_REGION = [450, 300, 650, 620]
OUT_DIR = "data/debug"
NUM_FRAMES = 30  # more frames to see pose dynamics


def cam_crop_to_full(pred_cam, bbox, img_h, img_w, fov_deg=60.0):
    pred_cam = pred_cam.copy()
    pred_cam[[0, 2]] *= -1
    s, tx, ty = pred_cam
    bc = np.array([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2])
    bs_raw = max(bbox[2]-bbox[0], bbox[3]-bbox[1])
    focal = img_h / (2 * math.tan(math.radians(fov_deg / 2)))
    bs = bs_raw * s + 1e-8
    tz = 2 * focal / bs
    cx = 2 * (bc[0] - img_w/2) / bs
    cy = 2 * (bc[1] - img_h/2) / bs
    return np.array([tx + cx, ty + cy, tz]), focal


def render_mesh(vertices, cam_t, focal_length, frame_bgr, faces):
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

    scene = pyrender.Scene(bg_color=[0,0,0,0], ambient_light=(0.3,0.3,0.3))
    scene.add(mesh)
    camera_pose = np.eye(4); camera_pose[:3, 3] = camera_translation
    scene.add(pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length, cx=w/2, cy=h/2, zfar=1e12),
              pose=camera_pose)

    for theta, phi in zip([np.pi/6]*3, [0, 2*np.pi/3, 4*np.pi/3]):
        xp, yp, zp = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
        z = np.array([xp, yp, zp]); z /= np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0: x = np.array([1, 0, 0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        mat = np.eye(4); mat[:3, :3] = np.c_[x, y, z]
        scene.add_node(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0), matrix=mat))

    renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
    color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()

    valid = (depth > 0).astype(np.float32)[:, :, None]
    color_f = color[:, :, :3].astype(np.float32) / 255.0
    output = color_f * valid + image * (1 - valid)
    return (output * 255).astype(np.uint8)


def run_variant(est, faces, clip, label, patch_fn=None):
    """Run inference on NUM_FRAMES frames, optionally patching behavior."""
    print(f"\n=== Variant: {label} ===")

    if patch_fn:
        patch_fn(True)  # apply patch

    cap = cv2.VideoCapture(clip)
    fps = cap.get(cv2.CAP_PROP_FPS)
    tracked_bbox = TARGET_REGION[:]

    out_path = f"{OUT_DIR}/mesh_{label}.mp4"
    w_out = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_out = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w_out, h_out))

    for i in range(NUM_FRAMES):
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        dets = detect_persons_cached(rgb, threshold=0.4)
        chosen = track_person(dets, tracked_bbox)
        if chosen:
            tracked_bbox = chosen
        else:
            chosen = [0, 0, w, h]

        result = est.predict(rgb, chosen, auto_detect=False)
        cam_t, focal = cam_crop_to_full(result['pred_camera'], result['bbox'], h, w)
        vis_rgb = render_mesh(result['pred_vertices'], cam_t, focal, frame, faces)
        vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
        writer.write(vis_bgr)

        if i % 10 == 0:
            v = result['pred_vertices']
            span = np.max(v, axis=0) - np.min(v, axis=0)
            print(f"  Frame {i}: height={span[1]:.3f}m")

    cap.release()
    writer.release()
    print(f"  Saved: {out_path}")

    if patch_fn:
        patch_fn(False)  # revert patch


def main():
    est = SAM3DBodyEstimator('/tmp/sam3d-mlx-weights-v2/')
    faces = np.array(est.model.head_pose.faces)
    model = est.model

    # --- Patch functions ---

    # BUG-5: CLIFF normalization
    import sam3d_mlx.batch_prep as bp
    _original_cliff = None

    def patch_cliff(enable):
        nonlocal _original_cliff
        if enable:
            _original_cliff = bp.get_cliff_condition

            def cliff_original(bbox, image_shape, focal_length=None):
                bbox = np.array(bbox, dtype=np.float32)
                H, W = image_shape
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                bw = bbox[2] - bbox[0]
                bh = bbox[3] - bbox[1]
                crop_size = max(bw, bh) * 1.2
                cx_norm = (cx - W / 2.0) / (W / 2.0)
                cy_norm = (cy - H / 2.0) / (H / 2.0)
                crop_ratio = crop_size / max(H, W)
                return np.array([cx_norm, cy_norm, crop_ratio], dtype=np.float32)

            bp.get_cliff_condition = cliff_original
        else:
            bp.get_cliff_condition = _original_cliff

    # BUG-6: Ray downsampling
    _original_condition_rays = None

    def patch_rays(enable):
        nonlocal _original_condition_rays
        if enable:
            _original_condition_rays = model.apply_ray_conditioning.__func__

            def rays_stride(self, image_features, rays):
                B, H_p, W_p, C = image_features.shape
                patch_size = self.config.patch_size
                rays_down = rays[:, ::patch_size, ::patch_size, :]
                ones = mx.ones((*rays_down.shape[:-1], 1))
                rays_3d = mx.concatenate([rays_down, ones], axis=-1)
                from sam3d_mlx.model import fourier_encode
                rays_flat = rays_3d.reshape(B, -1, 3)
                rays_encoded = fourier_encode(rays_flat)
                rays_encoded = rays_encoded.reshape(B, H_p, W_p, 99)
                combined = mx.concatenate([image_features, rays_encoded], axis=-1)
                combined_flat = combined.reshape(B, H_p * W_p, -1)
                conditioned = self.ray_cond_emb(combined_flat)
                return conditioned.reshape(B, H_p, W_p, C)

            import types
            model.apply_ray_conditioning = types.MethodType(rays_stride, model)
        else:
            import types
            model.apply_ray_conditioning = types.MethodType(_original_condition_rays, model)

    # BUG-7: LayerNorm32 -> plain LayerNorm
    # This one is structural — can't easily monkey-patch. Skip for now.

    # --- Run variants ---

    # A: current code (all fixes)
    run_variant(est, faces, CLIP, "A_all_fixes")

    # B: revert CLIFF only
    run_variant(est, faces, CLIP, "B_original_cliff", patch_cliff)

    # C: revert concat order (put estimate first, cliff last — the original)
    _original_forward = None

    def patch_concat(enable):
        nonlocal _original_forward
        if enable:
            _original_forward = model.__class__.__call__

            original_call = _original_forward

            def patched_call(self, *args, **kwargs):
                # Temporarily swap concat in the model
                # We patch by modifying the cliff_condition to put it last
                # Actually easier: just swap cliff and estimate in the input
                old_concat = mx.concatenate

                call_count = [0]
                def patched_concat(arrays, axis=0):
                    # The init_input concat is the one with a (B,3) and (B,522)
                    if (len(arrays) == 2 and
                        hasattr(arrays[0], 'shape') and hasattr(arrays[1], 'shape') and
                        arrays[0].shape[-1] == 3 and arrays[1].shape[-1] == 522):
                        # Swap order: put estimate first, cliff last
                        return old_concat([arrays[1], arrays[0]], axis=axis)
                    return old_concat(arrays, axis=axis)

                mx.concatenate = patched_concat
                try:
                    result = original_call(self, *args, **kwargs)
                finally:
                    mx.concatenate = old_concat
                return result

            model.__class__.__call__ = patched_call
        else:
            model.__class__.__call__ = _original_forward

    run_variant(est, faces, CLIP, "C_original_concat", patch_concat)

    print("\n=== Done. Check data/debug/mesh_*.mp4 ===")


if __name__ == "__main__":
    main()
