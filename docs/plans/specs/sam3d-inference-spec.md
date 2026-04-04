---
name: sam3d-inference
phase: 1
sprint: 2
parent: data-model
depends_on: [data-model, video-pipeline]
status: implemented
created: 2026-02-16
updated: 2026-04-04
---

# SAM 3D Inference Spec

Core 3D reconstruction: takes per-pitch video clips, runs SAM 3D Body on MPS, outputs mesh + skeleton data stored in SQLite + .npz.

## Decision: PyTorch/MPS

PyTorch on Apple's MPS backend gives identical output to CUDA at ~670ms/frame on M3 Max. The MLX port exists but is 2.4x slower with lower quality. No cloud GPU needed for batch analysis.

| Metric | PyTorch/MPS | MLX Port |
|--------|------------|----------|
| Median per frame | 670-748ms | 1773ms |
| Throughput | 1.1-1.5 fps | 0.5 fps |
| Mesh quality | Full fidelity | Approximated |

## Implementation

### Single-Video Inference (scripts/run_pytorch_video.py)

```bash
python scripts/run_pytorch_video.py -i video.mp4 --mode mesh     # 3D mesh overlay
python scripts/run_pytorch_video.py -i video.mp4 --mode skeleton  # skeleton lines
python scripts/run_pytorch_video.py -i video.mp4 --mode full      # 4-panel comparison
```

Modes: skeleton (fast), mesh (pyrender overlay), mesh+skeleton, full (original + skeleton + mesh + side view).

Pipeline per frame:
1. FasterRCNN MobileNetV3 person detection on MPS (largest bbox = pitcher)
2. `SAM3DBodyEstimator.process_one_image()` with detected bbox
3. Outputs: `pred_vertices (18439,3)`, `pred_keypoints_2d (70,2)`, `pred_cam_t (3,)`, `focal_length`, `pred_pose`, `pred_shape`
4. Render mesh via pyrender (offscreen OpenGL, macOS CGL)

### Batch Inference (scripts/batch_inference.py)

```bash
python scripts/batch_inference.py --game-pk 813024              # all clips for a game
python scripts/batch_inference.py --play-id 02ec65f0-...        # single pitch
python scripts/batch_inference.py --game-pk 813024 --reprocess  # re-run existing
```

Loads model once, iterates clips from PitchDB, stores results:
- Mesh data → `data/meshes/{game_pk}/{play_id[:8]}.npz`
- Metadata update → `pitches.db` (mesh_path, num_frames, inference_time_ms)

### Storage Format (.npz per pitch)

```python
vertices:     (T, 18439, 3)  float32  # full mesh per frame
joints_mhr70: (T, 70, 3)    float32  # MHR70 skeleton joints per frame
pose_params:  (T, 136)      float64  # SMPL pose per frame
shape_params: (45,)         float64  # body shape (constant)
cam_t:        (T, 3)        float32  # camera translation per frame
focal_length: ()            float64  # for re-rendering
```

~74MB per pitch clip (370 frames). ~3.8GB for a full game (52 pitches).

### Model Weights

- Checkpoint: `/tmp/sam3d-weights/model.ckpt`
- MHR body model: `/tmp/sam3d-weights/assets/mhr_model.pt`
- SAM 3D Body source: `/tmp/sam-3d-body/` (cloned from facebookresearch/sam-3d-body)
- Person detector: torchvision FasterRCNN MobileNetV3 (downloaded on first run)

## Validated

- Ohtani WS Game 7 pitch 1: 370 frames, 246s total, 670ms/frame, mesh stored in .npz
- Ohtani full clip (957 frames, 30fps): 836s total, 748ms/frame median
- Person detection isolates pitcher from batter/umpire/coach in broadcast footage
- MHR70 skeleton pairs from `sam_3d_body.metadata.mhr70.pose_info`

## Not Implemented (deferred)

- Kalman smoothing on joint trajectories (spec'd originally, not yet needed)
- MoGe2 camera estimation (using default FOV)
- VRAM/batch management (processing frame-by-frame, no batching needed on MPS)

## Files

| File | Purpose | Status |
|------|---------|--------|
| scripts/run_pytorch_video.py | Single-video SAM 3D Body inference with viz | implemented |
| scripts/batch_inference.py | Batch inference across clips → PitchDB storage | implemented |
| backend/app/data/pitch_db.py | MeshData storage/retrieval (.npz) | implemented |
| backend/app/pipeline/inference.py | Pipeline-integrated inference class | planned |
| backend/app/pipeline/smoothing.py | Kalman filter for temporal smoothing | planned |

## Dependencies

- Upstream: data-model (PitchDB, MeshData), video-pipeline (downloaded clips)
- Downstream: feature-extraction consumes joints_mhr70 from .npz
