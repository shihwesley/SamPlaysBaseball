# SamPlaysBaseball

[![License: CC BY-NC 4.0](https://img.shields.io/badge/license-CC%20BY--NC%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
![Status: Planning Complete](https://img.shields.io/badge/status-planning%20complete-yellow)
![Python](https://img.shields.io/badge/python-3.11+-blue)

Single-camera pitcher mechanics analyzer that turns bullpen clips, broadcast footage, or phone video into full 3D biomechanical breakdowns. Built for MLB player development staff who want motion-capture-grade analysis without an 8-camera KinaTrax setup.

```mermaid
flowchart LR
    A[Video Input] --> B[SAM 3D Body]
    B --> C[127-Joint Skeleton + Mesh]
    C --> D[Feature Extraction]
    D --> E[6 Analysis Modules]
    E --> F[Dashboard + Reports]
```

## Table of Contents

- [Why This Exists](#why-this-exists)
- [Features](#features)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Project Status](#project-status)
- [Key Research](#key-research)
- [License](#license)

## Why This Exists

KinaTrax is the industry standard for pitcher biomechanics. It achieves ~25mm accuracy but requires 8 synchronized cameras permanently installed in a stadium. That means no bullpen sessions, no MiLB ballparks without the hardware, and no analyzing phone footage a scout took at a showcase.

SAM 3D Body hits 54.8mm MPJPE from a single camera. That's roughly 2x the error, but good enough for the relative comparisons and trend detection that actually drive player development decisions. This project trades absolute precision for accessibility.

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

The repo ships with pre-computed analysis data. No GPU, no model downloads -- just the dashboard with real results.

```bash
docker compose up
```

Open `http://localhost:3000`. Historical legend comparisons are included.

For full pipeline usage (requires GPU):

```bash
docker compose --profile gpu up
```

Drop a video file into the upload interface and processing begins automatically.

## How It Works

Video frames go through SAM 3D Body (DINOv3-H+ backbone), which outputs a 127-joint skeleton plus full mesh per frame. SAM-Body4D handles temporal consistency with built-in Kalman smoothing. The Motion-Hierarchy Representation decouples skeleton from surface mesh, giving clean joint angle measurements without surface noise.

Feature extraction pulls biomechanical metrics from the skeleton sequence. Those feed into the six analysis modules, which run independently and produce structured results. The dashboard renders everything with 3D mesh replay (React Three Fiber) and interactive Plotly charts.

## Tech Stack

| Layer | Tech | Why |
|-------|------|-----|
| Frontend | Next.js 14 + Tailwind | Dark theme matches baseball video rooms |
| 3D Visualization | React Three Fiber | Component model, React integration |
| Charts | Plotly | Interactive hover/zoom, not static images |
| Backend | FastAPI + WebSocket | Real-time progress during video processing |
| 3D Reconstruction | SAM 3D Body / SAM-Body4D | 127 joints vs 25 (OpenPose) or 33 (MediaPipe). Full mesh with hand/foot decoders. |
| Storage | SQLite + Parquet | SQLite for queries, Parquet for numpy arrays |
| ML | scikit-learn, XGBoost, ruptures | Interpretable models. SHAP for explainability. |
| Deployment | Docker Compose | One command, portable |

## Project Status

Planning complete, implementation starting. 18 specs across 4 phases:

| Phase | What | Specs |
|-------|------|-------|
| 1 -- Foundation | Data model, video pipeline, 3D inference | 3 |
| 2 -- Analysis Engine | All 6 biomechanics modules + injury risk + Statcast | 8 |
| 3 -- Interface | API layer, 3D visualization, dashboard, scouting reports | 4 |
| 4 -- Polish | Demo mode, historical legends | 2 |

Full specs in [`docs/plans/specs/`](docs/plans/specs/). Dependency graph and sprint groupings in the [manifest](docs/plans/manifest.md).

## Key Research

- **Validation data:** Driveline OpenBiomechanics dataset -- 100K+ pitches of motion capture data
- **Comparable work:** PitcherNet (Waterloo/Orioles) uses SMPL with 22 joints; this project uses SAM 3D Body's 127
- **Normative ranges:** Stride length 77-87% of height, hip-shoulder separation 35-60 deg, max external rotation ~170 deg
- **Injury research:** Fatigue-related mechanical breakdown carries a 36x surgery risk increase (ASMI)
- **Correction potential:** ~50% of mechanical flaws are correctable through targeted movement modification

## License

SAM 3D Body is licensed CC BY-NC 4.0. This project is for portfolio/demonstration purposes.
