# SamPlaysBaseball

[![License: CC BY-NC 4.0](https://img.shields.io/badge/license-CC%20BY--NC%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
![Status: Active Development](https://img.shields.io/badge/status-active%20development-green)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![MLX](https://img.shields.io/badge/MLX-Apple%20Silicon-orange)

Single-camera pitcher mechanics analyzer that turns bullpen clips, broadcast footage, or phone video into full 3D biomechanical breakdowns. Built for MLB player development staff who want motion-capture-grade analysis without an 8-camera KinaTrax setup.

```mermaid
flowchart LR
    A[Video Input] --> B[SAM 3D Body / MLX]
    B --> C[127-Joint Skeleton + 18K Mesh]
    C --> D[Feature Extraction]
    D --> E[6 Analysis Modules]
    E --> F[Dashboard + Reports]
```

## Table of Contents

- [Why This Exists](#why-this-exists)
- [What Works Today](#what-works-today)
- [Features](#features)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [MLX Port](#mlx-port)
- [Tech Stack](#tech-stack)
- [Project Status](#project-status)
- [Key Research](#key-research)
- [License](#license)

## Why This Exists

KinaTrax is the industry standard for pitcher biomechanics. It achieves ~25mm accuracy but requires 8 synchronized cameras permanently installed in a stadium. That means no bullpen sessions, no MiLB ballparks without the hardware, and no analyzing phone footage a scout took at a showcase.

SAM 3D Body hits 54.8mm MPJPE from a single camera. That's roughly 2x the error, but good enough for the relative comparisons and trend detection that actually drive player development decisions. This project trades absolute precision for accessibility.

## What Works Today

- **3D body mesh estimation** on Apple Silicon via MLX (~490ms/frame on M3 Max 36GB)
- **Full numerical parity** with PyTorch/CUDA reference (<0.001mm vertex error across 18,439 vertices)
- **Video pipeline** -- frame-by-frame inference with skeleton overlay rendering
- **Statcast integration** -- per-pitch video clips linked to Baseball Savant tracking data
- **Height-based mesh scaling** -- roster height corrects mesh scale for real-world measurements
- **Batch inference** -- process full game clips, store meshes alongside pitch metadata

## Features

Six analysis modules, each targeting a specific question coaches ask:

- **Baseline Comparison** -- mechanical consistency against a pitcher's own history across outings
- **Pitch Tipping Detection** -- mechanical tells between pitch types (Random Forest + XGBoost, SHAP explainability)
- **Fatigue Tracking** -- gradual and sudden breakdown within games (Bayesian Online Changepoint Detection)
- **Command Analysis** -- mechanical patterns correlated with pitch location and miss patterns
- **Arm Slot Drift** -- release point and arm angle changes over time (27% of pitchers shift 5+ degrees year-over-year)
- **Timing Analysis** -- tempo consistency, hip-shoulder separation, and sequencing

Built on top of those:

- **Injury Risk Indicator** -- composite score from fatigue, arm slot drift, and timing anomalies. Called an "indicator," not a "predictor" -- honest framing matters.
- **Statcast Integration** -- mechanics to outcomes. "Your slider lost 3 inches of sweep because your hip-shoulder separation dropped 8 degrees."
- **AI Scouting Reports** -- LLM-generated one-pagers in the language scouts and coaches actually use.
- **Historical Legends Mode** -- analyze pitchers from archival footage. Single-camera 3D reconstruction makes this possible for the first time.

## Quick Start

### Run inference on Apple Silicon (no GPU required)

```bash
# Install dependencies
pip install mlx safetensors pillow numpy

# Convert weights (one-time)
python -m sam3d_mlx.convert_weights \
    --checkpoint /path/to/model.ckpt \
    --mhr-model /path/to/assets/mhr_model.pt \
    --output /tmp/sam3d-mlx-weights/

# Single image → 3D mesh
python -m sam3d_mlx --image photo.jpg --output mesh.obj

# Video → skeleton overlay
python -m sam3d_mlx.video --input pitch.mp4 --output pitch_overlay.mp4
```

### Python API

```python
from sam3d_mlx.generate import SAM3DPredictor

predictor = SAM3DPredictor.from_pretrained("/tmp/sam3d-mlx-weights")
result = predictor.predict(image_rgb, bbox=[x1, y1, x2, y2])

# result["pred_vertices"]      -> (18439, 3) mesh
# result["pred_keypoints_3d"]  -> (70, 3) keypoints
# result["pred_camera"]        -> (3,) weak-perspective camera
```

## How It Works

Video frames go through SAM 3D Body (DINOv3-H+ backbone, 32 layers, 1280d), which outputs a 127-joint skeleton plus full 18,439-vertex mesh per frame. The MHR (Meta Human Representation) body model runs forward kinematics, blend shapes, pose correctives, and linear blend skinning entirely in MLX -- no PyTorch at inference.

Feature extraction pulls biomechanical metrics from the skeleton sequence. Those feed into the six analysis modules, which run independently and produce structured results. The dashboard renders everything with 3D mesh replay and interactive charts.

## MLX Port

The SAM 3D Body model runs natively on Apple Silicon through a pure MLX reimplementation. This was contributed upstream to [mlx-vlm](https://github.com/Blaizzy/mlx-vlm/pull/922), following the SAM 3/3.1 porting pattern.

Three critical bugs were found and fixed during the port to achieve numerical parity:

| Bug | Impact | Fix |
|-----|--------|-----|
| `parameter_limits` applied at inference | ~17mm FK error, distorted poses | Removed -- JIT model skips it at inference |
| Scale formula `1 + dof` instead of `exp(dof * ln(2))` | Wrong joint scaling | Matched JIT's exponential formula |
| Pose correctives input: 889D raw DOFs instead of 750D 6D rotations | Broken corrective shapes | Use rotation features, not raw parameters |

Result: <0.001mm vertex error vs PyTorch across all 18,439 vertices.

| Metric | PyTorch/MPS | MLX | Improvement |
|--------|------------|-----|-------------|
| Inference (median) | 667ms | 488ms | 1.37x faster |
| Model load | 9.7s | 5.0s | 2.0x faster |
| FPS | 1.5 | 2.1 | -- |

See [`sam3d_mlx/README.md`](sam3d_mlx/README.md) for full architecture and API docs.

## Tech Stack

| Layer | Tech | Why |
|-------|------|-----|
| 3D Reconstruction | SAM 3D Body | 127 joints + 18K mesh vs 25 (OpenPose) or 33 (MediaPipe) |
| Inference | MLX (Apple Silicon) | Native Metal, no CUDA needed, 2 FPS on M3 Max |
| Inference (alt) | PyTorch/MPS | Fallback for non-MLX environments |
| Backend | FastAPI + WebSocket | Real-time progress during video processing |
| Frontend | Next.js 14 + Tailwind | Dark theme matches baseball video rooms |
| 3D Visualization | React Three Fiber | Component model, React integration |
| Charts | Plotly | Interactive hover/zoom, not static images |
| Data | Baseball Savant / Statcast | Per-pitch tracking data linked to video |
| Storage | SQLite + Parquet | SQLite for queries, Parquet for numpy arrays |
| ML | scikit-learn, XGBoost, ruptures | Interpretable models. SHAP for explainability. |
| Deployment | Docker Compose | One command, portable |

## Project Status

18 specs across 4 phases. Phase 1 foundation work is underway:

| Phase | What | Status |
|-------|------|--------|
| 1 -- Foundation | Data model, video pipeline, 3D inference | **MLX inference complete**, Statcast pipeline working |
| 2 -- Analysis Engine | All 6 biomechanics modules + injury risk + Statcast | Planned |
| 3 -- Interface | API layer, 3D visualization, dashboard, scouting reports | Planned |
| 4 -- Polish | Demo mode, historical legends | Planned |

### Completed

- MLX port of SAM 3D Body with full PyTorch parity (3 body model bugs fixed)
- Weight converter (PyTorch .ckpt + JIT .pt to MLX safetensors)
- Video inference pipeline with skeleton overlay
- Statcast integration (per-pitch clips + tracking data)
- Height-based mesh scaling from roster data
- Batch inference across full game pitch sequences

Full specs in [`docs/plans/specs/`](docs/plans/specs/). Dependency graph and sprint groupings in the [manifest](docs/plans/manifest.md).

## Key Research

- **Validation data:** Driveline OpenBiomechanics dataset -- 100K+ pitches of motion capture data
- **Comparable work:** PitcherNet (Waterloo/Orioles) uses SMPL with 22 joints; this project uses SAM 3D Body's 127
- **Normative ranges:** Stride length 77-87% of height, hip-shoulder separation 35-60 deg, max external rotation ~170 deg
- **Injury research:** Fatigue-related mechanical breakdown carries a 36x surgery risk increase (ASMI)
- **Correction potential:** ~50% of mechanical flaws are correctable through targeted movement modification

## License

SAM 3D Body is licensed CC BY-NC 4.0. This project is for portfolio/demonstration purposes.
