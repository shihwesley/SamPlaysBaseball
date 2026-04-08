# SamPlaysBaseball

[![License: CC BY-NC 4.0](https://img.shields.io/badge/license-CC%20BY--NC%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
![Status: Active Development](https://img.shields.io/badge/status-active%20development-green)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![MLX](https://img.shields.io/badge/MLX-Apple%20Silicon-orange)

**A post-game film-review tool for pitching analysts.** Type a question, get a 3D side-by-side comparison of deliveries with the mechanical differences quantified and rendered.

> *"I think he was tipping his slider in the sixth."*
> Type that into the query bar. The tool fetches per-pitch Statcast clips, runs SAM 3D Body inference locally on Apple Silicon, compares slider deliveries against fastball deliveries from the same outing, and shows you the in-plane body-posture differences frame by frame.

```mermaid
flowchart LR
    A[Natural-language query] --> B[Local Gemma 4 E4B parser]
    B --> C[Baseball Savant per-pitch clips]
    C --> D[SAM 3D Body / MLX inference]
    D --> E[Phase-aligned delivery comparison]
    E --> F[3D viewer + diagnostic narrative]
```

## Table of Contents

- [Why This Exists](#why-this-exists)
- [The Headline Use Case: Tipping Confirmation](#the-headline-use-case-tipping-confirmation)
- [What's Actually Reliable](#whats-actually-reliable-read-validationmd-for-the-honest-version)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [MLX Port](#mlx-port)
- [Tech Stack](#tech-stack)
- [Project Status](#project-status)
- [Key Research](#key-research)
- [License](#license)

## Why This Exists

KinaTrax is the industry standard for pitcher biomechanics. It achieves ~25mm joint accuracy but requires eight synchronized cameras permanently installed in a stadium. That means:

- No bullpen sessions outside the stadium
- No MiLB ballparks without the install
- No archived broadcast footage
- No phone clips a scout took at a college showcase
- No analyzing video on your laptop after the game

SAM 3D Body achieves **54.8mm MPJPE** from a single camera. That's roughly 2x the error of KinaTrax, but it's good enough for the **relative** comparisons that drive most post-game film-review decisions ("did this delivery look different from his last delivery?"), and it works on any video that exists.

This project is what you build when you want post-game film review on a laptop with no hardware install and no cloud bills.

## The Headline Use Case: Tipping Confirmation

The thing this tool is best at — and the thing that motivated the current scope — is **post-game tipping confirmation**.

Here's the workflow:

1. During a game, a coach (yours or the opposing team's) notices the pitcher might be tipping a pitch type. Maybe his glove sets differently for the slider. Maybe his shoulder posture in the stretch is different. The information is *visual and live*, not recorded.
2. After the game, an analyst types into the dashboard:

   > *"Compare Darvish's slider deliveries to his fastball deliveries from Game 7."*

3. The tool fetches the per-pitch Savant clips for that outing, runs SAM 3D Body inference on each delivery, groups by pitch type, and uses phase-aligned comparison to surface the **measurable in-plane body-posture differences** between his slider deliveries and his fastball deliveries.
4. The analyst gets a 3D side-by-side view (slider on the left, fastball on the right, ghost overlay), a frame-aligned timeline scrubber, joint click for any body part, and an LLM-generated narrative that names the specific mechanical difference and which frame it appears in.
5. The coach now has something concrete to show the player: *"Here. Frame 24. Your glove is 4 cm lower on slider than on fastball. The other team is reading it."*

This works because tipping tells are typically **in-plane body-posture differences** — glove height, shoulder set, hand placement, pre-delivery rhythm. Those are exactly the signals where a single-camera 3D body model is at its most reliable. (See [VALIDATION.md](VALIDATION.md) for why this is the case.)

The tool is **not** trying to predict tipping that no one has noticed. It's trying to **confirm and quantify** what a human observer already saw.

## What's Actually Reliable (read VALIDATION.md for the honest version)

| Capability | Confidence |
|------------|------------|
| Pairwise delivery comparison (same camera, same pitcher, same outing) | High |
| Phase detection and timing | High |
| In-plane joint angles (knee flex, trunk lean, stride direction) | Medium-high |
| Tipping confirmation (post-game, body-posture differences) | Medium-high |
| Absolute arm slot in degrees | Medium — use deltas, not absolute values |
| Release point depth (z) | Medium — single-camera depth is the weak axis |
| Velocity / spin / pitch trajectory | **Not measured** — use Statcast / Hawk-Eye |
| Elbow torque, UCL stress, injury prediction | **Not measured** — see Future Biomechanics Work in VALIDATION.md |

The single-camera approach trades absolute precision for accessibility. Read [VALIDATION.md](VALIDATION.md) for the full breakdown of what this tool can and cannot tell you, the literature-cited error bars, and how the numbers were validated.

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

### Run the full app

```bash
# Backend
cd backend && pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend (separate terminal)
cd frontend && npm install && npm run dev

# Open http://localhost:3000/analyze
```

## How It Works

A natural-language query goes through a local Gemma 4 E4B parser (via mlx-vlm) which extracts a structured `AnalysisQuery` (pitcher, pitch types, comparison mode, inning ranges). The orchestrator resolves the pitcher, fetches per-pitch Baseball Savant clips, runs SAM 3D Body inference (MLX backend by default, ~490ms/frame on M3 Max) on any uncached pitches, and stores meshes plus joint sequences in SQLite + .npz.

`compare_deliveries()` does phase-aligned comparison: it normalizes the deliveries by detected mechanical phases (windup, leg lift, foot strike, MER, release), aligns them frame-by-frame, and produces spatial divergence metrics, kinetic chain comparison, and per-joint trajectory deltas. The diagnostic engine (Gemma 4 E4B locally, with provider fallback to Claude/OpenAI/Ollama) takes that comparison and writes a scout-readable narrative naming the specific differences and the frames they appear at.

The dashboard is a 3-column War Room layout: left is the React Three Fiber 3D viewer (mesh, ghost overlay, timeline scrubber, joint click for any body part, camera presets), middle is the diagnostic report, right is Statcast context. The whole flow runs locally with zero API cost when using the Gemma backend.

## MLX Port

The SAM 3D Body model runs natively on Apple Silicon through a pure MLX reimplementation. This was contributed upstream to [mlx-vlm](https://github.com/Blaizzy/mlx-vlm/pull/922), following the SAM 3/3.1 porting pattern.

Three critical bugs were found and fixed during the port to achieve numerical parity:

| Bug | Impact | Fix |
|-----|--------|-----|
| `parameter_limits` applied at inference | ~17mm FK error, distorted poses | Removed — JIT model skips it at inference |
| Scale formula `1 + dof` instead of `exp(dof * ln(2))` | Wrong joint scaling | Matched JIT's exponential formula |
| Pose correctives input: 889D raw DOFs instead of 750D 6D rotations | Broken corrective shapes | Use rotation features, not raw parameters |

Result: <0.001mm vertex error vs PyTorch across all 18,439 vertices.

| Metric | PyTorch/MPS | MLX | Improvement |
|--------|------------|-----|-------------|
| Inference (median) | 667ms | 488ms | 1.37x faster |
| Model load | 9.7s | 5.0s | 2.0x faster |
| FPS | 1.5 | 2.1 | — |

See [`sam3d_mlx/README.md`](sam3d_mlx/README.md) for full architecture and API docs.

## Tech Stack

| Layer | Tech | Why |
|-------|------|-----|
| 3D Reconstruction | SAM 3D Body | 127 joints + 18K mesh vs 25 (OpenPose) or 33 (MediaPipe) |
| Inference | MLX (Apple Silicon) | Native Metal, no CUDA needed, 2 FPS on M3 Max |
| Inference (alt) | PyTorch/MPS | Fallback for non-MLX environments |
| Local LLM | Gemma 4 E4B via mlx-vlm | Query parsing + diagnostic narrative, fully local, zero API cost |
| LLM fallback | Claude / OpenAI / Ollama | Provider-agnostic interface |
| Backend | FastAPI + WebSocket | Real-time progress during video processing |
| Frontend | Next.js 14 + Tailwind | War Room dark theme |
| 3D Visualization | React Three Fiber | Component model, React integration |
| Charts | Plotly | Interactive hover/zoom |
| Data | Baseball Savant / Statcast | Per-pitch tracking data linked to video |
| Storage | SQLite + .npz | SQLite for queries, .npz for mesh and joint arrays |
| Comparison | Phase-aligned delivery comparison | Custom — see `backend/app/analysis/compare_deliveries.py` |

## Project Status

**Current state:** End-to-end query-driven pipeline working on Ohtani 2024 data. 263 backend tests passing. Health 82/100. Frontend builds with 0 type errors. Browser QA on all 4 pages. War Room dashboard live. MLX inference is the default backend.

**What's in scope right now:**

| Component | Status |
|-----------|--------|
| MLX SAM 3D Body port + numerical parity | **shipped** |
| Per-pitch Savant clip fetching | **shipped** |
| Pitch database (SQLite + .npz) | **shipped** |
| Phase-aligned delivery comparison | **shipped** |
| Tipping confirmation (repositioned: post-game, in-plane body-posture diff) | **shipped** |
| Arm slot drift tracking (within-outing consistency) | **shipped** |
| Baseline comparison | **shipped** |
| Timing analysis | **shipped** |
| Local Gemma 4 E4B query parser | **shipped** |
| Diagnostic engine (LLM narrative) | **shipped** |
| Query orchestrator + API | **shipped** |
| War Room dashboard | **shipped** |
| 3D viewer (R3F): mesh, ghost overlay, scrubber, joint click | **shipped** |
| Blender render pipeline for demo videos | **in progress** |

**Deferred (see [VALIDATION.md](VALIDATION.md) → Future Biomechanics Work):**

| Component | Why deferred |
|-----------|--------------|
| `injury_risk` | Combines unvalidated signals into a medical-adjacent score. Needs marker-mocap validation + cohort data. |
| `fatigue_tracking` | Changepoint detection on sparse broadcast clips produces too many false positives. Needs dense in-game data. |
| `command_analysis` | Requires Hawk-Eye trajectory data per pitch. The input side is unsolved. |

The deferred modules' code is preserved in `backend/app/analysis/` so the work can be revived when the validation backing exists.

Full specs in [`docs/plans/specs/`](docs/plans/specs/). Dependency graph and sprint groupings in the [manifest](docs/plans/manifest.md).

## Key Research

- **Validation pathway:** [Driveline OpenBiomechanics dataset](https://github.com/drivelineresearch/openbiomechanics) — 100K+ pitches of marker-mocap data. The validation plan is to compare this tool's joint angles, release point, and arm slot against marker-mocap ground truth. Identified, not yet run.
- **Comparable work:** PitcherNet (Waterloo/Orioles) uses SMPL with 22 joints; this project uses SAM 3D Body's 127.
- **Normative ranges (population, not derived by this tool):** Stride length 77-87% of height, hip-shoulder separation 35-60°, max external rotation ~170° (ASMI).
- **Injury research (context, not claims):** Fatigue-related mechanical breakdown carries a 36x surgery risk increase (ASMI). This is population-level research, not something this tool can derive.
- **MLX port reference:** Followed the SAM 3/3.1 porting pattern from `mlx-vlm`. Upstream PR: [mlx-vlm#922](https://github.com/Blaizzy/mlx-vlm/pull/922).

## License

SAM 3D Body is licensed CC BY-NC 4.0. This project is for portfolio and demonstration purposes. Analyze footage you have the right to analyze — broadcast clips are owned by their rights holders, and this tool's local-inference design specifically avoids redistributing them.
