---
name: mlx-inference
phase: 3
sprint: 1
parent: mlx-port-manifest
depends_on: [mlx-weight-converter, mlx-backbone, mlx-decoder, mlx-mhr-head]
status: draft
created: 2026-04-03
---

# MLX Inference Pipeline

End-to-end SAM 3D Body inference on MLX. No PyTorch in the model forward pass.
Person detection remains external (torchvision/YOLO as a preprocessing step).

## Requirements

- Load safetensors weights into MLX model
- Process single image: RGB numpy → 3D mesh + joints
- Process video: frame-by-frame with person detection
- Support both "body" and "full" inference modes
- Batch preparation (crop, normalize, camera intrinsics) in MLX/numpy
- Person detection is external (torchvision or YOLO, separate process — out of scope for this port)
- Mixed precision: backbone + decoder in float16, norms compute in float32, MHR FK in float32
- Output format matches PyTorch: pred_vertices, pred_keypoints_3d/2d, joint_coords

## Acceptance Criteria

- [ ] `sam3d_mlx.load_model(path)` loads weights and returns ready model
- [ ] `model.predict(image, bbox)` returns same dict keys as PyTorch estimator
- [ ] Video pipeline processes 30fps 720p at > 2 fps on M-series
- [ ] Peak memory < 6GB for single-image body mode
- [ ] CLI: `python -m sam3d_mlx --image photo.jpg --output mesh.obj`

## Files

| File | Action |
|------|--------|
| `sam3d_mlx/__init__.py` | create |
| `sam3d_mlx/model.py` | create (SAM3DBody top-level) |
| `sam3d_mlx/estimator.py` | create (image/video processing) |
| `sam3d_mlx/__main__.py` | create (CLI entry point) |
| `sam3d_mlx/batch_prep.py` | create (crop, normalize, intrinsics) |

## Tasks

1. Wire backbone + decoder + mhr_head into SAM3DBody top-level module
2. Port batch preparation (prepare_batch) to numpy/MLX
3. Implement SAM3DBodyEstimator (process_one_image equivalent)
4. Add video processing loop with frame extraction
5. Create CLI with argparse (image, video, output modes)
