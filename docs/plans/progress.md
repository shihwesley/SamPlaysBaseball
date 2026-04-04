# Progress Log

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
| data-model | 1 | 1 | needs-revision | 2026-04-04 |
| statcast-integration | 1 | 1 | needs-revision | 2026-04-04 |
| pitch-fetcher | 1 | 1 | **implemented** | 2026-04-04 |
| video-pipeline | 1 | 2 | needs-revision | 2026-04-04 |
| sam3d-inference | 1 | 2 | needs-revision | 2026-04-04 |
| pitch-matcher | 1 | 2 | designed | 2026-04-04 |
| feature-extraction | 2 | 1 | draft | 2026-02-16 |
| mesh-export | 2 | 1 | draft | 2026-02-16 |
| baseline-comparison | 2 | 2 | draft | 2026-02-16 |
| tipping-detection | 2 | 2 | draft | 2026-02-16 |
| fatigue-tracking | 2 | 2 | draft | 2026-02-16 |
| command-analysis | 2 | 2 | draft | 2026-02-16 |
| arm-slot-drift | 2 | 2 | draft | 2026-02-16 |
| timing-analysis | 2 | 2 | draft | 2026-02-16 |
| injury-risk | 2 | 3 | draft | 2026-02-16 |
| ai-scouting-reports | 3 | 1 | draft | 2026-02-16 |
| api-layer | 3 | 1 | draft | 2026-02-16 |
| 3d-visualization | 3 | 2 | draft | 2026-02-16 |
| dashboard-ui | 3 | 2 | draft | 2026-02-16 |
| blender-render | 3 | 2 | draft | 2026-02-16 |
| historical-legends | 4 | 1 | draft | 2026-02-16 |
| demo-mode | 4 | 1 | draft | 2026-02-16 |
| mlx-weight-converter | MLX-1 | 1 | draft | 2026-04-03 |
| mlx-backbone | MLX-1 | 2 | draft | 2026-04-03 |
| mlx-decoder | MLX-2 | 1 | draft | 2026-04-03 |
| mlx-mhr-head | MLX-2 | 2 | draft | 2026-04-03 |
| mlx-inference | MLX-3 | 1 | draft | 2026-04-03 |
| mlx-validation | MLX-3 | 1 | draft | 2026-04-03 |

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
