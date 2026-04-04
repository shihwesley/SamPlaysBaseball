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
