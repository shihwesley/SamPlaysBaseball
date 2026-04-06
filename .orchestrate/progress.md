# SamPlaysBaseball Orchestration Progress

## Phase 2 Sprint 2 — 6 Analysis Modules
- Date: 2026-04-03
- Commit: 285b8e7 (feat), 786b6d4 (merge)
- Status: completed
- Tests: 41 passing, 0 failing
- Modules delivered:
  - baseline.py — z-score deviation, severity tiers
  - tipping.py + shap_utils.py — RandomForest + SHAP feature importance
  - fatigue.py + changepoint.py — rolling window, CUSUM changepoint detection
  - command.py — release point scatter, command scoring
  - arm_slot.py — within-game drift, GMM bimodal detection
  - timing.py + energy.py — kinetic chain validation, energy decomposition
- Deferred: 0 (requirements.txt additions added but xgboost/shap/ruptures optional at runtime)

## Phase 2 Sprint 3 — Composite Injury Risk Indicator
- Date: 2026-04-03
- Commit: d0966de (feat), 8b97ca6 (merge)
- Status: completed
- Tests: 36 passing, 0 failing
- Files delivered:
  - backend/app/analysis/injury_risk.py — InjuryRiskCalculator with ASMI weights
  - backend/tests/test_injury_risk.py — 36 tests covering all acceptance criteria
- Risk factors: fatigue_mechanics_drift (0.30), kinetic_chain_disruption (0.25),
  arm_slot_instability (0.20), workload (0.15), hip_shoulder_separation_deficit (0.10)
- Features: traffic light (green/yellow/orange/red), per-factor % contribution,
  trend tracking with streak detection, positional average comparison, report generation
- Deferred: 0

## Phase 3 Sprint 2 — mesh-export + 3d-visualization + dashboard-ui + blender-render
- Date: 2026-04-03
- Commit: 7cc0901 (feat), fast-forward merge to main
- Status: completed
- Tests: 6 passing, 3 skipped (pygltflib not installed; tests correct, skip conditional)
- Frontend build: Next.js 16.2.2 Turbopack — clean (0 type errors)
- Files delivered:
  - backend/app/export/ — GLBExporter (morph targets), GroundPlaneAligner (PCA), MLBMound, ComparisonGLBBuilder
  - backend/tests/test_glb_export.py — 9 tests (6 pass, 3 skip pygltflib)
  - frontend/ — Next.js 16 App Router, dark theme
    - 3D: MoundScene, PitcherMesh, SkeletonOverlay, DeviationColoring, GhostOverlay, SplitSync, Stroboscope, TimelineScrubber, CameraPresets
    - Pages: /, /pitcher/[id], /pitcher/[id]/outing/[outingId], /compare, /upload
    - Charts: AngleTimeSeries, ReleasePointScatter, FatigueCurve, KineticChain, RadarDeviation, ArmSlotHistory, TippingImportance
    - UI: Sidebar, PitcherCard
    - Lib: api.ts, mesh-loader.ts
  - scripts/blender/ — scene_setup, camera_presets, render_pitch, render_comparison, render_stroboscope, batch_render
- Review: P0=0, P1=0, P2=7, P3=9 (all deferred)
- Deferred: 11 items (memoization, camera preset updates, pygltflib tests, gltfpack quantization, frontend tests)

## Phase 4 Sprint 1 — demo-mode + historical-legends
- Date: 2026-04-03
- Commit: abe7767 (feat), 08229d9 (merge)
- Status: completed
- Tests: 39 passing, 2 failing (pre-existing e2e failures, unrelated to this phase)
- Files delivered:
  - demo/__init__.py, generate_synthetic.py, process_samples.py, launcher.py
  - demo/docker-compose.yml, Dockerfile.api, Dockerfile.frontend
  - demo/README.md, TALKING_POINTS.md
  - demo/legends/process_legends.py, legends/talking_points.json, legends/README.md
  - demo/data/ + demo/data/legends/ + demo/results/ placeholders
- Synthetic data: 50 pitches across 3 pitchers (sam_torres, jake_kim, demo_pitcher_01)
- All 6 analysis modules, baselines, joint data — GPU-free, runs with python3 + numpy only
- Legends: Rivera, Pedro, Kershaw, Maddux, Randy Johnson with biomechanical talking points

## Phase 3 Sprint 1 — api-layer + ai-scouting-reports
- Date: 2026-04-03
- Commit: 1bc756b (feat), merge to main
- Status: completed
- Tests: 34 passing, 0 failing
- Files delivered:
  - backend/app/main.py — FastAPI app, CORS, lifespan, StorageLayer init
  - backend/app/api/ — 10 modules: deps, models, routes, pitchers, pitches, analysis, upload, compare, websocket, demo, reports
  - backend/app/reports/ — generator, templates, llm, pdf
  - StorageLayer.load_analysis_by_pitcher() added to avoid N+1 queries
- Endpoints: 17 routes covering pitchers, pitches, analysis (7 modules), upload/jobs, compare, WebSocket, demo, reports + PDF
- Reports: pitcher/outing/pitch-type report types; LLM narrative via Claude API; template fallback; PDF via reportlab
- Fixes applied: path traversal sanitization on upload, N+1 query on analysis endpoints
- Deferred (P2/P3): WebSocket test, demo code dedup, datetime.utcnow deprecation, PDF temp cleanup
