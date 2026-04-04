---
name: sam3d-inference
phase: 1
sprint: 2
parent: data-model
depends_on: [data-model, video-pipeline]
status: in-progress
created: 2026-02-16
updated: 2026-04-03
---

# SAM 3D Inference Spec

Core 3D reconstruction: takes extracted frames, runs SAM 3D Body, outputs MHR joint + mesh data.

## Decision: PyTorch/MPS (not MLX)

The MLX port was completed but produces lower-quality output due to simplified parameter limits and body model approximations. PyTorch running on Apple's MPS backend gives identical output to CUDA at ~650ms/frame on Apple Silicon. No cloud GPU needed.

- **Primary path:** PyTorch/MPS on Mac (Apple Silicon)
- **MLX port:** Retained for reference/benchmarking, not used in production pipeline
- **TurboQuant:** No longer relevant (was MLX-only optimization)

## Requirements

- Run Meta's SAM 3D Body model (DINOv3-H+ variant, 840M params) via PyTorch/MPS
- Person detection via Faster R-CNN (torchvision, runs on same MPS device)
- Frame-by-frame inference with optional Kalman smoothing
- Output per-frame: mesh vertices (18439, 3), keypoints (70, 3), joint coords (127, 3), camera (3,)
- Output per-frame: face topology (36874, 3) — constant across frames

## Acceptance Criteria

- [ ] SAM 3D Body loads from checkpoint on MPS device
- [ ] Person detection per frame, largest bbox used
- [ ] Kalman smoothing on joint trajectories for temporal consistency
- [ ] Fixed-camera override when intrinsics provided
- [ ] Output: PitchData with `joints_3d (T, 127, 3)`, `keypoints_3d (T, 70, 3)`, `vertices (T, 18439, 3)`, `camera (T, 3)`, `faces (36874, 3)`
- [ ] Processing speed: ~1.5 fps on M-series Mac
- [ ] Video pipeline: `scripts/run_pytorch_video.py` supports skeleton, mesh, and full modes

## Technical Approach

Load SAM 3D Body from `/tmp/sam3d-weights/model.ckpt` with MHR body model from `/tmp/sam3d-weights/assets/mhr_model.pt`. Uses SAM3DBodyEstimator from reference code directly. Person detection via torchvision Faster R-CNN MobileNetV3 on same MPS device.

Mesh rendering via pyrender (offscreen OpenGL) for visualization modes. Keypoints follow MHR70 convention (not COCO-17) — see `sam_3d_body.metadata.mhr70` for joint ordering.

## Files

| File | Purpose |
|------|---------|
| backend/app/pipeline/inference.py | SAM3DInference class, SAM-Body4D wrapper |
| backend/app/pipeline/smoothing.py | Kalman filter + Butterworth fallback |
| backend/app/pipeline/camera.py | MoGe2 wrapper + fixed camera override |
| backend/app/pipeline/gpu.py | VRAM management, batch size auto-tuning |
| backend/tests/test_inference.py | Inference tests (mock model for CI) |

## Tasks

1. Set up SAM 3D Body model loading from HuggingFace
2. Implement SAM-Body4D video pipeline wrapper
3. Build Kalman smoothing fallback for frame-by-frame mode
4. Implement camera estimation (MoGe2 + fixed override)
5. Build GPU memory manager with auto batch sizing
6. Write output aggregation to PitchData format

## Dependencies

- Upstream: data-model (PitchData format), video-pipeline (extracted frames)
- Downstream: feature-extraction consumes joint data
