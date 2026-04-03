---
name: sam3d-inference
phase: 1
sprint: 2
parent: data-model
depends_on: [data-model, video-pipeline]
status: draft
created: 2026-02-16
---

# SAM 3D Inference Spec

Core 3D reconstruction: takes extracted frames, runs SAM-Body4D, outputs MHR joint data.

## Requirements

- Integrate Meta's SAM 3D Body model (DINOv3-H+ variant, 840M params)
- Use SAM-Body4D for video temporal consistency (preferred over raw frame-by-frame)
- Fallback to frame-by-frame + Kalman smoothing when SAM-Body4D unavailable
- Camera estimation via MoGe2, with override for known camera intrinsics
- Output PitchData objects with full joint/pose/shape data per pitch

## Acceptance Criteria

- [ ] SAM 3D Body loads from HuggingFace (`facebook/sam-3d-body-dinov3`)
- [ ] SAM-Body4D pipeline processes video frames with temporal consistency
- [ ] Kalman smoothing fallback produces clean joint trajectories
- [ ] Fixed-camera override skips MoGe2 when intrinsics provided
- [ ] Batch inference supported (configurable batch size based on VRAM)
- [ ] Output: PitchData with `joints_3d (T, 127, 3)`, `body_pose (T, 136)`, `shape_params (45,)`
- [ ] GPU memory management: graceful handling of OOM, auto batch size reduction
- [ ] Processing speed logged: frames/sec, pitches/hour

## Technical Approach

Load SAM 3D Body via the official `notebook.utils.setup_sam_3d_body` API. For SAM-Body4D: SAM 3 video segmentation for identity-consistent masks → SAM 3D Body per frame with mask guidance → Kalman smoothing on MHR pose parameters. Fixed body shape from first visible frame (shape doesn't change between pitches).

Batch processing with padding for SAM-Body4D's 2x speedup. GPU memory: start at batch_size=4, halve on OOM, retry.

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
