# Path C Progress: Running SAM 3D Body on macOS

> Date: 2026-04-03
> Status: **Full inference working on CPU and MPS (Apple GPU).** 6.2x speedup with MPS.

## What Works

- Pixi 0.66.0 installed at ~/.pixi/bin/pixi
- Python 3.12 venv via pixi at /tmp/sam-3d-body/.pixi/envs/default/
- pymomentum-cpu 0.1.108 installed via conda-forge (fully working)
- PyTorch 2.10.0 with MPS available
- All SAM 3D Body deps installed (roma, einops, timm, etc.)
- MHR body model tested standalone — produces 18,439 vertices + 127 joints
- SAM 3D Body model loaded: 1,285M params
- DINOv3-H+ backbone loads via torch.hub
- Model weights downloaded to /tmp/sam3d-weights/ (2.1GB checkpoint + 696MB MHR JIT)
- MHR assets at /tmp/MHR/assets/
- **Full inference pipeline working (both "body" and "full" modes)**
- **MPS acceleration working with mixed-device strategy**

## Performance

| Device | Body Mode | Full Mode | Notes |
|--------|-----------|-----------|-------|
| CPU    | 11.5s     | ~35s (est)| All on CPU |
| MPS    | 1.1-1.9s  | 4.8s      | Backbone+decoder on GPU, MHR on CPU |
| Speedup| **6.2x**  | ~7x (est) | |

## Output Per Person

- `pred_vertices`: (18439, 3) — full body mesh
- `pred_keypoints_3d`: (70, 3) — 3D joint positions
- `pred_keypoints_2d`: (70, 2) — projected 2D joints
- `pred_joint_coords`: (127, 3) — all skeleton joint coordinates
- Plus: cam_t, pose_raw, global_rot, body_pose, hand_pose, scale, shape, face params

## Patches Applied

1. `/tmp/sam-3d-body/sam_3d_body/sam_3d_body_estimator.py:160`
   - Changed: `recursive_to(batch, "cuda")` → `recursive_to(batch, str(self.device))`

2. `/tmp/sam-3d-body/sam_3d_body/models/meta_arch/sam3d_body.py`
   - All `.cuda()` calls → `.to(self.device)` (5 instances via sed)
   - `recursive_to(batch_lhand, "cuda")` → `recursive_to(batch_lhand, str(self.device))`
   - `recursive_to(batch_rhand, "cuda")` → `recursive_to(batch_rhand, str(self.device))`

3. `/tmp/sam-3d-body/sam_3d_body/utils/dist.py:25-31` (NEW)
   - `recursive_to` now downcasts float64 → float32 when target is MPS

4. `/tmp/sam-3d-body/sam_3d_body/models/heads/mhr_head.py` (NEW)
   - Added `_mhr_force_cpu` flag and `ensure_mhr_on_cpu()` method
   - `mhr_forward` shuttles inputs to CPU for JIT MHR, outputs back to device
   - Reason: JIT MHR uses float64 internally (baked into TorchScript), MPS doesn't support float64

5. `/tmp/sam-3d-body/sam_3d_body/build_models.py:38-41` (NEW)
   - After `model.to(device)`, calls `ensure_mhr_on_cpu()` on both head_pose and head_pose_hand
   - Keeps JIT MHR on CPU while rest of model runs on MPS

## Next Steps

1. ~~Fix post-processing error~~ ✓ (was already working after CUDA→device patches)
2. ~~Try MPS backend for GPU acceleration~~ ✓ (6.2x speedup)
3. Test with a real baseball player image (need person detector or manual bbox)
4. Visualize the mesh overlay on the image (needs renderer — trimesh/pyrender or custom)
5. Profile to find remaining bottlenecks
6. Consider: can we batch multiple frames for video processing?
7. Integrate with the SamPlaysBaseball pipeline

## Architecture Notes

The mixed-device strategy works because:
- DINOv3-H+ backbone (840M params, ViT) dominates compute → runs on MPS
- Transformer decoder → runs on MPS
- MHR body model (JIT) is lightweight (FK + skinning) but needs float64 → stays on CPU
- Data shuttle overhead is negligible: ~200 float params to CPU, ~18K verts back to MPS

## Environment Commands

```bash
# Activate pixi environment
export PATH="$HOME/.pixi/bin:$PATH"
cd /tmp/sam-3d-body

# Run Python in pixi env
pixi run python your_script.py

# Key paths
# Model weights: /tmp/sam3d-weights/model.ckpt
# MHR JIT: /tmp/sam3d-weights/assets/mhr_model.pt
# MHR assets: /tmp/MHR/assets/
# SAM 3D Body source: /tmp/sam-3d-body/
# mlx-vlm source: /tmp/mlx-vlm/
# MHR source: /tmp/MHR/
```

## HuggingFace Access
- Logged in as: shihwesley
- Token: set via HF_TOKEN env var
- Model access: granted for facebook/sam-3d-body-dinov3
