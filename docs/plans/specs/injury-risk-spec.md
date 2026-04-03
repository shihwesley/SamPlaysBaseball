---
name: injury-risk
phase: 2
sprint: 3
parent: feature-extraction
depends_on: [fatigue-tracking, arm-slot-drift, timing-analysis]
status: draft
created: 2026-02-16
---

# Injury Risk Score Spec

Combines biomechanical data from multiple analysis modules into a single injury risk estimate per pitcher.

## Requirements

- Aggregate fatigue, arm slot, and timing data into a composite risk score
- Weight factors by research-backed injury correlations
- Track risk over time (per-outing, per-season)
- Actionable breakdown: which factors are driving the risk score

## Acceptance Criteria

- [ ] Composite risk score: 0-100 scale (0 = low risk, 100 = immediate concern)
- [ ] Risk factors weighted by ASMI research:
  - Fatigue-related mechanical breakdown (36x surgery risk)
  - Elbow valgus torque proxy (from timing/arm angle)
  - Kinetic chain disruption (trunk-arm timing gap)
  - Arm slot instability (variance within game)
  - Workload (pitch count, recovery days)
- [ ] Per-factor contribution: "fatigue accounts for 40% of current risk"
- [ ] Traffic light display: green (0-30), yellow (31-60), orange (61-80), red (81-100)
- [ ] Trend tracking: risk score per outing, flag increasing trends
- [ ] Comparison: pitcher's risk vs positional average
- [ ] Report: top 3 risk drivers + recommended mechanical adjustments

## Technical Approach

Weighted linear combination of normalized risk factors. Weights from ASMI and published research:
- Fatigue mechanics drift: 0.30 (36x correlation is the strongest signal)
- Kinetic chain disruption: 0.25 (arm-dominant delivery = high elbow stress)
- Arm slot instability: 0.20 (unpredictable mechanics = unpredictable loads)
- Workload: 0.15 (pitch count + days since last outing)
- Hip-shoulder separation deficit: 0.10 (forces compensatory arm acceleration)

Each factor normalized to 0-1 range, weighted sum × 100 = risk score.

This isn't a predictive model (we don't have injury outcome data to train on). It's a risk *indicator* based on published biomechanics research. Honest framing matters — call it "risk indicator" not "predictor."

## Files

| File | Purpose |
|------|---------|
| backend/app/analysis/injury_risk.py | InjuryRiskCalculator class |
| backend/tests/test_injury_risk.py | Risk calculation tests |

## Tasks

1. Implement risk factor extraction from analysis module outputs
2. Build weighted composite risk score calculator
3. Implement trend tracking (risk over time)
4. Generate injury risk report with per-factor breakdown

## Dependencies

- Upstream: fatigue-tracking, arm-slot-drift, timing-analysis
- Downstream: api-layer, ai-scouting-reports
