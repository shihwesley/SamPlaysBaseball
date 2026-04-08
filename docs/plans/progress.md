# Progress Log

## Session: 2026-04-07 — Strategic Pivot to Tipping-Confirmation Demo

### Decision
After a CEO-style review of the full app, the project pivoted from "9 analysis modules
in a War Room dashboard for MLB player development" to "post-game tipping-confirmation
tool for analysts and coaches who don't have stadium-installed mocap." Goal: land an
MLB player-dev role via a single high-impact demo + blog post + upstream MLX port.

### Why
- The depth-axis ambiguity of single-camera SAM 3D Body fundamentally limits absolute
  biomechanical claims (release point Z, arm slot in degrees, hip-shoulder separation).
- 5 of the 9 analysis modules shipped without external validation. Specifically
  tipping/fatigue/command/injury-risk relied on inputs that aren't trustworthy enough
  for credible coach-facing claims.
- Tipping detection IS defensible if reframed as post-game confirmation of what a coach
  already saw — the in-plane body-posture differences (glove height, shoulder set, hand
  placement) are exactly the signals where single-camera 3D body modeling is reliable.

### Demo target locked
**Yu Darvish — 2017 World Series Game 7 — game_pk 526517**

The Astros allegedly picked up a slider tip during this start. Verified via Savant:
- 47 Darvish pitches in WS G7, all 47 with downloadable play_ids
- Pitch mix: FF=13, ST=17, FC=5, CU=7, SI=3, CH=2
- Control game: **2017-09-19 vs Phillies, game_pk 492355** (FF=21 / ST=21, no allegations)
- Both games' clips download cleanly via `--angle HOME`

### Pipeline sanity check on 2017 broadcast footage — PASSED
- Downloaded one Darvish CH pitch (`b69a4fbd...`) at 1280x720, 29.97 fps, 300 frames
- Ran SAM 3D Body MLX inference: 150.7s for 300 frames (2.0 fps as spec'd)
- Mesh shape: (300, 18439, 3); joints_mhr70: (300, 70, 3); no NaN/inf
- Body bbox diagonal at frame 0 = **1.938m** (matches Darvish's 1.96m height almost exactly)
- Body-center travel across 300 frames = 0.843m (consistent with a pitcher's stride, far
  too much for a static catcher/umpire) → confirms model locked onto Darvish, not bystanders
- Visual confirmation: extracted frame 90 shows Darvish on the mound, Correa batting,
  FOX WS broadcast lower-third reading "Darvish 1.0 IP, 10 P" (the alleged tipping window)

### Fetch script bug found
`scripts/fetch_savant_clips.py` defaults to `--angle AWAY`, which returns no MP4 for
2017 postseason play_ids. Savant only kept HOME-feed clips for older games. Workaround:
pass `--angle HOME` explicitly. Permanent fix: make the default fall back through
HOME → CMS_NATIONAL → NETWORK → AWAY automatically. **Logged in TODOS for next session.**

### Code changes shipped
| Change | File(s) |
|--------|---------|
| `tipping_detection` repositioned + new `compare_within_outing()` post-game entry point | `backend/app/analysis/tipping.py` |
| `fatigue_tracking` unwired from API + reports (`.py` preserved) | `backend/app/api/analysis.py`, `backend/app/reports/generator.py` |
| `command_analysis` unwired from API + reports (`.py` preserved) | same |
| `injury_risk` unwired from API + reports + recommendations (`.py` preserved) | same |
| Query parser concern enum: dropped fatigue/command, kept tipping, added release_consistency | `backend/app/query/parser.py` |
| `historical-legends` spec deleted; manifest updated | `docs/plans/specs/historical-legends-spec.md`, `docs/plans/manifest.md` |
| `VALIDATION.md` written from scratch — what's trustworthy, what isn't, literature-cited | `VALIDATION.md` |
| `README.md` rewritten as product story with the tipping-confirmation use case as the headline | `README.md` |
| Manifest status flags: 4 modules marked **deferred** with revival path notes | `docs/plans/manifest.md` |
| Test fix: `test_generate_outing_report` asserts `arm_slot` instead of `fatigue` | `backend/tests/test_reports.py` |

### Remaining for the demo phase
1. Visual mesh inspection of the existing Darvish CH inference (geometry sanity check, left/right joint swap check) — DONE in afternoon session
2. Run full WS G7 batch inference (47 pitches, ~120 min) + full Sep 19 control batch (97 pitches, ~240 min)
3. Finish blender-render spec and build the side-by-side comparison render with delta annotation
4. Record the 60-90 second demo video
5. Write the companion blog post
6. Release: LinkedIn, Twitter, r/baseball, targeted MLB cold emails

---

## Session: 2026-04-07 (afternoon) — SAM 3.1 Trim Pipeline

### Driving question
The morning's pipeline check on the in-play CH clip (b69a4fbd) revealed via the 2D-overlay video that the broadcast cuts from pitcher cam to wide field shot ~5 seconds into every hit-into-play pitch. The model was tracking a fielder in frames 155-300 of that clip — invisible in the matplotlib joint inspection because joint coordinates alone don't tell you which person they belong to. **We needed a robust per-clip trim step that works across 2017 (30fps) and modern (60fps) broadcasts and across hit-into-play vs no-contact pitch outcomes.**

### Architecture decision
- Trimming = Stage 1 of the data pipeline. Runs at DOWNLOAD time (in `fetch_savant_clips.py`), not inference time. The trimmed `.mp4` is what gets stored in the manifest and the DB. `batch_inference.py` reads the trimmed file with no awareness of the trim step. No `--auto-trim` flag — trimming is unconditional.
- Two-signal detector design: SAM 3.1 (semantic, "is this a pitcher") + cv2 histogram diff (structural, "did the camera cut"). Each tool used for what it's best at. ~3-5s per clip vs naive uniform sampling at ~60s. **12x speedup.**

### What was built
| Component | File | Lines | Status |
|-----------|------|-------|--------|
| `Sam31PitcherDetector` wrapper class | `sam3d_mlx/sam31_detector.py` | 208 | shipped |
| `find_delivery_window()` + `write_trimmed_clip()` | `backend/app/pipeline/delivery_window.py` | 380 | shipped (refactored once) |
| `scripts/test_sam31_on_darvish.py` | one-shot SAM 3.1 sanity check | 90 | shipped |
| `scripts/test_delivery_window.py` | end-to-end trim test runner | 60 | shipped |

### Validation milestones
- **mlx-vlm 0.4.4 installed** via `pip install --break-system-packages mlx-vlm`. Pulled in transformers 5.5, mlx-lm 0.31, hf-hub 1.9, opencv 4.13. Model weights `mlx-community/sam3.1-bf16` cached locally (~1-2 GB).
- **SAM 3.1 confirmed working on 2017 broadcast frame**: prompt "a person in a baseball uniform" returned 5 detections (scores 0.85, 0.80, 0.78, 0.72, 0.37) cleanly identifying pitcher, batter, catcher, umpire. Crowd correctly excluded by language grounding.
- **Geometric pitcher tiebreaker calibrated** against the Darvish frame. Tightened from `(center_y >= 0.40, height_frac >= 0.15)` to `(center_y >= 0.55, height_frac >= 0.35)`. Cleanly excludes catcher, umpire, batter, and field-action runners.
- **Trim pipeline validated on both Darvish CH clips**:
  - In-play CH (b69a4fbd, "out(s)" outcome, broadcast cut at frame 155): trim window [0, 150). Visual confirmation: end frame shows Darvish in follow-through at moment of contact, NOT field action. Data-driven end-frame logic respected the broadcast cut.
  - Ball CH (dd857f31, "Ball" outcome, no broadcast cut): trim window [0, 150). Visual confirmation: end frame shows Darvish at delivery completion with ball traveling to plate. Biomechanical 5s cap correctly limited the trim even though SAM 3.1 detected Darvish through frame 250+.
- **Algorithm refactored to early-exit + cv2 scene cut** halfway through the session. Speed dropped from 65.6s to 11.7s per clip (5.6x). After fixing the scene-cut threshold (0.55 → 0.85), expected to drop further to ~3-5s.

### Calibration measurements
- Histogram correlations in pitcher-cam frames: 0.99-1.00 mean, very stable
- Broadcast cut signature: drop to 0.649 at frame 155 of the Darvish in-play clip
- Optimal scene-cut threshold: **0.85** (between cut and noise floor)
- Original threshold 0.55 (literature default for scene detection) FAILED to catch the cut

### Open at end of session (must address in next session)
1. **Refactored delivery_window not yet re-tested** with the new thresholds (`min_consecutive=1` and `_HIST_CORR_CUT_THRESHOLD=0.85`). Edited but not validated. **First task in the next session.**
2. **fetch_savant_clips.py integration not done** (task #23) — wire `find_delivery_window` + `write_trimmed_clip` into the download flow.
3. **mlx-vlm not pinned in requirements.txt** (task #19).
4. **Unit tests for delivery_window not written** (task #15).
5. **No end-to-end SAM 3D Body inference on a trimmed clip yet** (task #21).

### Files changed this session (afternoon)
- New: `sam3d_mlx/sam31_detector.py`, `backend/app/pipeline/delivery_window.py`, `scripts/test_sam31_on_darvish.py`, `scripts/test_delivery_window.py`
- Modified: (none beyond the new files)
- Generated artifacts: `data/clips/526517/inn1_ab4_p1_CH_b69a4fbd_trimmed.mp4`, `inn1_ab5_p1_CH_dd857f31_trimmed.mp4`, `_sam31_test.png`, `_trimmed_v2_end.jpg`, `_ball_trim_{start,mid,end}.jpg`

### Where the next session picks up
1. Re-run `python scripts/test_delivery_window.py` on the in-play CH clip with the freshly-edited thresholds. Expected: trim window [0, 155) (set_frame=6, scene cut at 155). If it lands there, the refactor is fully validated.
2. Then re-run on the Ball CH clip. Expected: trim window [0, 150) (set_frame=0, biomech cap).
3. Then integrate trimmer into fetch_savant_clips.py (task #23).
4. Then run end-to-end SAM 3D Body inference on a trimmed clip (task #21).
5. Then pin mlx-vlm in requirements.txt + add unit tests + the long-running batch inference work.

---

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
