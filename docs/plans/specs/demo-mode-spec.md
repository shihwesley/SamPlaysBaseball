---
name: demo-mode
phase: 4
sprint: 1
parent: null
depends_on: [3d-visualization, dashboard-ui]
status: draft
created: 2026-02-16
---

# Demo Mode Spec

Pre-computed demo data + packaging so the entire app runs without a GPU. This is what you actually bring to a meeting.

## Requirements

- Process 3-5 sample pitchers from publicly available video
- Store all pipeline outputs (joint data, features, analysis results)
- Frontend loads demo data without backend processing
- One-command launch (no GPU, no model download)
- Documentation for running the demo

## Acceptance Criteria

- [ ] 3-5 sample pitchers processed, each with 20+ pitches across pitch types
- [ ] At least one pitcher with visible tipping (real or synthetic)
- [ ] At least one pitcher showing fatigue progression over an outing
- [ ] Pre-computed data: joint positions, features, all 6 analysis results, mesh vertices
- [ ] Static JSON/Parquet files in demo/data/ (no database needed)
- [ ] Frontend demo mode: toggle that reads from static files instead of API calls
- [ ] Docker compose or single script: `./demo.sh` launches frontend + lightweight API
- [ ] Total demo data size < 500MB (reasonable for git LFS or zip download)
- [ ] README with demo walkthrough: what to click, what to point out in a meeting
- [ ] Talking points document: what each analysis shows, what it means for player development

## Technical Approach

Process videos on a GPU machine ahead of time. Serialize everything to JSON (metadata/analysis) + Parquet (numpy arrays). The demo API is a stripped-down FastAPI that reads static files — no SAM 3D Body model loaded, no inference. Frontend detects demo mode via environment variable and adjusts data fetching.

Docker Compose with two services: frontend (Next.js static build) + api (lightweight Python serving static data). Total startup time < 30 seconds.

Include a "demo script" — a markdown file that walks through the presentation: "Start on the pitcher list, click on Pitcher A, show the tipping analysis, switch to the 3D view, rotate the mesh, then compare fastball vs changeup."

## Files

| File | Purpose |
|------|---------|
| demo/process_samples.py | Script to process sample videos (run on GPU machine) |
| demo/data/ | Pre-computed pitch data, features, analysis results |
| demo/results/ | Pre-computed analysis outputs |
| demo/launcher.py | Demo launch script |
| demo/docker-compose.yml | Docker Compose for demo deployment |
| demo/Dockerfile.api | Lightweight API container |
| demo/Dockerfile.frontend | Frontend container |
| demo/README.md | Demo setup and walkthrough |
| demo/TALKING_POINTS.md | Presentation guide for meetings |

## Tasks

1. Identify and acquire 3-5 sample pitcher videos (public domain / fair use)
2. Process sample videos through full pipeline (GPU required)
3. Serialize all outputs to static files (JSON + Parquet)
4. Build lightweight demo API (static file serving, no model)
5. Configure frontend demo mode toggle
6. Create Docker Compose deployment
7. Write demo walkthrough and talking points documentation

## Dependencies

- Upstream: 3d-visualization, dashboard-ui (full app must be built first)
- Downstream: none (final deliverable)
