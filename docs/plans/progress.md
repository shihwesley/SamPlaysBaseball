# Progress Log

## Session: 2026-04-05 (continued)

### Milestones
- **Mechanics diagnostic fully implemented:** All 6 steps from the design doc done
- **Query parser:** NL → AnalysisQuery via local Gemma 4 E4B (mlx-vlm) or Claude (fallback)
- **Orchestrator pipeline:** parse → resolve pitcher → fetch pitches → compare deliveries → Statcast aggregation → diagnostic report → GLB export
- **API endpoints:** POST /api/query (submit) + GET /api/query/{token}/status (poll)
- **Analyze dashboard:** War Room design, 3-column layout (40% viewer / 35% report / 25% stats)
- **3D interaction components:** JointSelector (raycasting click), MetricGraph (Plotly time-series), SpeedControl, FieldGeometry
- **Design direction approved:** War Room base + Lab Notebook spaciousness (via /design-shotgun)
- **QA passed:** 263 backend tests, 0 failures. Browser QA on all 4 pages. 2 bugs fixed.
- **Fully local LLM:** Both query parser and diagnostic engine run on Gemma 4 E4B via mlx-vlm. Zero API costs.

### What was built
| Component | File | Commit |
|-----------|------|--------|
| Query parser (local Gemma4) | backend/app/query/parser.py | d339a38 |
| Query orchestrator | backend/app/query/orchestrator.py | c9b7e2b |
| Query API endpoints | backend/app/api/query.py | c9b7e2b |
| Analyze dashboard page | frontend/src/app/analyze/page.tsx | a367a6b |
| QueryBar component | frontend/src/components/ui/QueryBar.tsx | a367a6b |
| ReportPanel component | frontend/src/components/ui/ReportPanel.tsx | a367a6b |
| MetricsPanel component | frontend/src/components/ui/MetricsPanel.tsx | a367a6b |
| StatcastPanel component | frontend/src/components/ui/StatcastPanel.tsx | a367a6b |
| JointSelector component | frontend/src/components/three/JointSelector.tsx | e262828 |
| MetricGraph component | frontend/src/components/three/MetricGraph.tsx | e262828 |
| SpeedControl component | frontend/src/components/three/SpeedControl.tsx | e262828 |
| FieldGeometry component | frontend/src/components/three/FieldGeometry.tsx | 48058f7 |
| War Room design styling | globals.css, layout.tsx, all panels | 8daa960 |
| QA fix: API key handling | backend/app/api/query.py | f66bed7 |
| QA fix: datetime deprecation | backend/app/api/upload.py | c0a014a |

### Validation
- 263 backend tests passing (pytest, 3.5s)
- Frontend builds with 0 type errors (tsc --noEmit)
- Browser QA: all 4 pages render correctly, query flow works end-to-end (error handling verified)
- Health score: 82/100

### Decisions
- War Room design direction with Lab Notebook spacing (approved via design-shotgun)
- Gemma 4 E4B is the default LLM for both parsing AND diagnostics (fully local)
- Claude API is cloud fallback only (requires ANTHROPIC_API_KEY env var)
- Mobile responsiveness deferred (desktop-first tool for MLB analysts)

---

## Session: 2026-04-05

### Milestones
- **MLX bugs fixed and validated:** 3 bugs in MHR body model corrected (parameter_limits, scale formula, pose correctives). Vertices now match PyTorch within 0.0001mm.
- **MLX now default backend:** batch_inference.py defaults to --backend mlx (~490ms/frame on M3 Max)
- **Phase 2 fully validated:** 112 tests passing across all feature extraction, analysis, and export modules
- **Delivery comparison built:** compare_deliveries() function with phase-normalized alignment, spatial divergence, kinetic chain comparison
- **Diagnostic report engine built:** Provider-agnostic DiagnosticEngine with Gemma4/Claude/OpenAI/Ollama backends
- **Mechanics diagnostic designed:** Full design doc for query-driven pitcher analysis tool (docs/plans/2026-04-05-mechanics-diagnostic-design.md)
- **All Phase 2 + MLX specs marked implemented**

### What was built
| Component | File | Commit |
|-----------|------|--------|
| MLX pred_shape/pred_pose output | sam3d_mlx/mhr_head.py, estimator.py | fbdacf9 |
| MLX backend for batch_inference | scripts/batch_inference.py | fbdacf9 |
| Backend-agnostic inference ABC | backend/app/pipeline/inference.py | fbdacf9 |
| Delivery comparison module | backend/app/analysis/compare_deliveries.py | fbdacf9 |
| Diagnostic report engine | backend/app/reports/diagnostic.py | b18ae44 |
| Normative biomechanical ranges | backend/app/reports/norms.py | b18ae44 |
| Mechanics diagnostic design doc | docs/plans/2026-04-05-mechanics-diagnostic-design.md | 1c21f36 |

### Validation
- Ran fresh MLX inference on ohtani_ws_g7_pitch1.mp4 (120 frames at 60fps)
- Feature extraction on post-fix MLX output shows real motion: elbow flex range 88 deg, angular vel 1914 deg/s
- Kinetic chain is sequential (pelvis → trunk → shoulder → elbow �� wrist)
- Pre-fix data was essentially static (0.5 deg variation) — confirmed by comparing file dates to bug fix commits

### Decisions
- MLX is the default inference backend (was PyTorch/MPS)
- Gemma 4 E4B via mlx-vlm for local multimodal diagnostic reports (vision + text, ~16GB)
- Provider-agnostic LLM interface — can swap to Claude API, Ollama, or any OpenAI-compatible endpoint
- On-demand inference with cache for MVP; batch pre-compute after games for production
- Hybrid dashboard: query bar (LLM-parsed) + traditional filters, 3D viewer + report + metrics

---

## Session: 2026-04-04

### Milestones
- **SAM 3D Body on MPS validated:** 1.1 fps PyTorch/MPS on M3 Max (2.4x faster than MLX port)
- **Baseball Savant pipeline built:** per-pitch clips with Statcast linkage, end-to-end
- **Pitch database built:** SQLite + .npz storage with Statcast enrichment
- **TODO-001 deferred:** GPU spike moved to VISION-007 (no local NVIDIA GPU)
- **Manifest and specs audited:** 4 specs marked needs-revision, 6 MLX specs added

### What was built
| Component | File | Commit |
|-----------|------|--------|
| Pitch fetcher | scripts/fetch_savant_clips.py | 7ed40d6 |
| Player search | backend/app/data/player_search.py | 7ed40d6 |
| Pitch database | backend/app/data/pitch_db.py | 7ed40d6 |
| Batch inference | scripts/batch_inference.py | 7ed40d6 |
| Pitch matcher design | docs/plans/2026-04-04-pitch-matcher-design.md | 7ed40d6 |

### Validation
- Fetched 52 Ohtani pitches from WS Game 7 (game_pk=813024)
- Ran SAM 3D Body on one clip: 370 frames, 670ms/frame, mesh stored
- Statcast enrichment: 50/52 pitches enriched with velocity, spin, movement, outcomes
- Full pipeline: search player → game log → download clips → inference → DB storage

### Decisions
- Baseball Savant per-pitch clips are the primary data source (pre-segmented, Statcast-linked)
- Cloud GPU not required for batch analysis — M3 Max handles the Ohtani MVP
- MLX port drops in priority (optimization, not a blocker)
- Pitch matcher (TODO-002) becomes a fallback for non-Savant video sources

---

## Session: 2026-04-03

### Milestones
- **CEO review:** Vision reframed to dual-mode (batch + live companion), Ohtani-first MVP
- **MPS validation runs:** Both PyTorch and MLX tested on ohtani_full.mp4 (957 frames)
- **MLX port eng review:** 6 specs created and reviewed, backbone params fixed

### Key findings
- Fast SAM 3D Body (arXiv:2603.15603): 10.9x speedup, makes real-time feasible
- SAM-Body4D: temporal consistency for video
- SAM4Dcap: video → biomechanics directly (needs NVIDIA GPU)
- fal.ai: $0.02/image for prototyping

---

## Session: 2026-02-16

### Milestones
- **Initial planning complete:** 18 specs across 4 phases, 7 sprints
- Created manifest, findings, progress files
- All specs in draft status

---

## Spec Status

| Spec | Phase | Sprint | Status | Last Updated |
|------|-------|--------|--------|-------------|
| data-model | 1 | 1 | **implemented** | 2026-04-04 |
| statcast-integration | 1 | 1 | **implemented** | 2026-04-04 |
| pitch-fetcher | 1 | 1 | **implemented** | 2026-04-04 |
| video-pipeline | 1 | 2 | **implemented** | 2026-04-04 |
| sam3d-inference | 1 | 2 | **implemented** | 2026-04-04 |
| pitch-matcher | 1 | 2 | designed | 2026-04-04 |
| feature-extraction | 2 | 1 | **implemented** | 2026-04-05 |
| mesh-export | 2 | 1 | **implemented** | 2026-04-05 |
| baseline-comparison | 2 | 2 | **implemented** | 2026-04-05 |
| tipping-detection | 2 | 2 | **implemented** | 2026-04-05 |
| fatigue-tracking | 2 | 2 | **implemented** | 2026-04-05 |
| command-analysis | 2 | 2 | **implemented** | 2026-04-05 |
| arm-slot-drift | 2 | 2 | **implemented** | 2026-04-05 |
| timing-analysis | 2 | 2 | **implemented** | 2026-04-05 |
| injury-risk | 2 | 3 | **implemented** | 2026-04-05 |
| ai-scouting-reports | 3 | 1 | **implemented** | 2026-04-05 |
| api-layer | 3 | 1 | **implemented** | 2026-04-05 |
| 3d-visualization | 3 | 2 | **implemented** | 2026-04-05 |
| dashboard-ui | 3 | 2 | **implemented** | 2026-04-05 |
| blender-render | 3 | 2 | draft | 2026-02-16 |
| historical-legends | 4 | 1 | draft | 2026-02-16 |
| demo-mode | 4 | 1 | draft | 2026-02-16 |
| mlx-weight-converter | MLX-1 | 1 | **implemented** | 2026-04-05 |
| mlx-backbone | MLX-1 | 2 | **implemented** | 2026-04-05 |
| mlx-decoder | MLX-2 | 1 | **implemented** | 2026-04-05 |
| mlx-mhr-head | MLX-2 | 2 | **implemented** | 2026-04-05 |
| mlx-inference | MLX-3 | 1 | **implemented** | 2026-04-05 |
| mlx-validation | MLX-3 | 1 | **implemented** | 2026-04-05 |

### Status legend
- **draft** — spec written, not yet implemented
- **needs-revision** — spec exists but implementation diverged; rewrite before orchestrating
- **designed** — design doc exists, not yet coded
- **implemented** — code exists and works, may or may not have a formal spec
- **complete** — implemented + tested + reviewed

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Phase 1 partially done via direct builds. 4 specs need revision before orchestrating. |
| Where am I going? | Feature extraction (Phase 2 Sprint 1) is next. |
| What's the goal? | Pitcher mechanics analyzer — Ohtani MVP from broadcast footage. |
| What have I learned? | findings.md (updated 2026-04-04) |
| What have I done? | Pipeline works end-to-end: fetch → infer → store → enrich. |
