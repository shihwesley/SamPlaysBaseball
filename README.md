# SamPlaysBaseball

A pitcher mechanics analyzer that turns single-camera video into full 3D biomechanical breakdowns. Built for MLB player development personnel.

Drop in a bullpen clip, broadcast footage, or phone video. The system reconstructs 3D body mechanics using Meta's SAM 3D Body model (127 joints, full mesh), runs six biomechanical analysis modules, scores injury risk, correlates with Statcast outcomes, and generates AI scouting reports in scout-readable language. Everything renders in an interactive dashboard with 3D mesh replay.

## What It Does

**Video in, analysis out.** The pipeline handles four video sources: bullpen cameras, broadcast feeds, phone recordings, and MiLB footage.

**Six analysis modules:**

- **Baseline Comparison** — track mechanical consistency against a pitcher's own baseline across outings
- **Pitch Tipping Detection** — find mechanical tells between pitch types using Random Forest + XGBoost with SHAP explainability
- **Fatigue Tracking** — detect gradual and sudden mechanical breakdown within games using Bayesian Online Changepoint Detection
- **Command Analysis** — correlate mechanical patterns with pitch location and miss patterns
- **Arm Slot Drift** — monitor release point and arm angle changes over time (27% of pitchers shift 5+ degrees year-over-year)
- **Timing Analysis** — measure tempo consistency, hip-shoulder separation, and sequencing

**Beyond the six:**

- **Injury Risk Indicator** — composite score from fatigue, arm slot drift, and timing anomalies. Called an "indicator," not a "predictor" — honest framing matters.
- **Statcast Integration** — closes the loop from mechanics to outcomes. "Your slider lost 3 inches of sweep because your hip-shoulder separation dropped 8 degrees."
- **AI Scouting Reports** — LLM-generated one-pagers that translate data into the language scouts and coaches actually use.
- **Historical Legends Mode** — analyze pitchers from archival footage. Single-camera 3D reconstruction makes this possible for the first time — KinaTrax didn't exist in 1995.

## How It Works

The core pipeline: video frames → SAM 3D Body (DINOv3-H+ backbone) → 127-joint skeleton + mesh → feature extraction → analysis modules → dashboard.

SAM-Body4D handles temporal consistency across frames with built-in Kalman smoothing. The Motion-Hierarchy Representation decouples skeleton from surface mesh, giving clean joint angle measurements without surface noise.

For context: the industry standard (KinaTrax) achieves ~25mm accuracy but requires 8 synchronized cameras. SAM 3D Body hits 54.8mm MPJPE from a single camera — good enough for the relative comparisons and trend detection this tool focuses on.

## Tech Stack

| Layer | Tech | Why |
|-------|------|-----|
| Frontend | Next.js 14 + Tailwind | Looks like a product, not a prototype. Dark theme matches baseball video rooms. |
| 3D Visualization | React Three Fiber | Component model, React integration, ecosystem |
| Charts | Plotly | Interactive, hover tooltips, zoom — not static Matplotlib images |
| Backend | FastAPI + WebSocket | Real-time progress feedback during video processing |
| 3D Reconstruction | SAM 3D Body / SAM-Body4D | 127 joints vs 25 (OpenPose) or 33 (MediaPipe). Full 3D mesh with hand/foot decoders for release point tracking. |
| Storage | SQLite + Parquet | SQLite for queries, Parquet for numpy arrays. Keeps the DB lean. |
| ML | scikit-learn, XGBoost, ruptures | Interpretable models over black boxes. SHAP for explainability. |
| Deployment | Docker Compose | One command, no dependencies, portable |

## Demo Mode

The repo ships with pre-computed analysis data so you can run the full dashboard without a GPU. Historical legend comparisons are included. One command:

```bash
docker compose up
```

No GPU. No model downloads. Just the dashboard with real analysis results.

## Project Status

**Planning complete, implementation starting.** 18 detailed specs across 4 phases:

| Phase | What | Specs |
|-------|------|-------|
| 1 — Foundation | Data model, video pipeline, 3D inference | 3 |
| 2 — Analysis Engine | All 6 biomechanics modules + injury risk + Statcast | 8 |
| 3 — Interface | API layer, 3D visualization, dashboard, scouting reports | 4 |
| 4 — Polish | Demo mode, historical legends | 2 |

Full spec files live in [`docs/plans/specs/`](docs/plans/specs/). The dependency graph and sprint groupings are in the [manifest](docs/plans/manifest.md).

## Target Architecture

```
SamPlaysBaseball/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry
│   │   ├── models/              # Pydantic/dataclass models
│   │   ├── pipeline/            # Video + SAM3D + feature extraction
│   │   ├── analysis/            # 6 analysis modules + injury risk
│   │   ├── data/                # Statcast integration
│   │   ├── reports/             # AI scouting report generation
│   │   └── api/                 # Route handlers
│   └── tests/
├── frontend/
│   ├── src/
│   │   ├── app/                 # Next.js pages
│   │   ├── components/
│   │   │   ├── three/           # 3D visualization
│   │   │   ├── charts/          # Plotly charts
│   │   │   └── ui/              # Shared components
│   │   └── lib/                 # API client, utilities
│   └── package.json
├── demo/
│   ├── data/                    # Pre-computed pitch data
│   ├── results/                 # Pre-computed analysis results
│   └── legends/                 # Historical pitcher analysis
└── docs/
    └── plans/                   # Specs and planning docs
```

## Key Research

- **Validation data:** Driveline OpenBiomechanics dataset — 100K+ pitches of motion capture data
- **Comparable work:** PitcherNet (Waterloo/Orioles) uses SMPL with 22 joints; this project uses SAM 3D Body's 127
- **Normative ranges:** Stride length 77-87% of height, hip-shoulder separation 35-60 deg, max external rotation ~170 deg
- **Injury research:** Fatigue-related mechanical breakdown carries a 36x surgery risk increase (ASMI)
- **Correction potential:** ~50% of mechanical flaws are correctable through targeted movement modification

## License

SAM 3D Body is licensed CC BY-NC 4.0. This project is for portfolio/demonstration purposes.
