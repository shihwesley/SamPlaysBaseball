---
name: fatigue-tracking
phase: 2
sprint: 2
parent: feature-extraction
depends_on: [feature-extraction]
status: draft
created: 2026-02-16
---

# Fatigue Tracking Spec

Tracks mechanical degradation across an outing. Research shows fatigue-related breakdown correlates with 36x increase in surgery risk.

## Requirements

- Establish "fresh" baseline from early-outing pitches (first 1-3 innings)
- Track key fatigue indicators pitch-by-pitch
- Changepoint detection: when does mechanical breakdown start?
- Compare fatigue patterns across outings
- Report severity with injury risk context

## Acceptance Criteria

- [ ] Rolling statistics: 5-pitch sliding window for all fatigue indicators
- [ ] Fatigue indicators tracked: arm slot angle, release point drift, hip-shoulder separation, stride length, trunk tilt at release, lead leg brace firmness
- [ ] Changepoint detection: Bayesian Online Changepoint Detection (BOCPD) or CUSUM
- [ ] Alert thresholds: configurable, default 1.5 SD from fresh baseline
- [ ] Cross-outing comparison: track fatigue onset pitch number across starts
- [ ] Fatigue curve visualization data: metric value vs pitch number
- [ ] Minimum data: 30+ pitches per outing
- [ ] Report: when breakdown started, which metrics drifted, magnitude, comparison to previous outings

## Technical Approach

From pitcher-mechanics.md: after 100 pitches, expect -14% internal rotation strength, -10% flexion strength. Observable in 3D data as arm slot drop, release point drift down and arm-side, decreased hip-shoulder separation, shorter stride, increased trunk tilt.

BOCPD (bayesian_changepoint package or ruptures library) for detecting the moment mechanics shift. Rolling Z-scores against fresh baseline for continuous monitoring. Cross-outing comparison stores fatigue onset pitch number per start.

## Files

| File | Purpose |
|------|---------|
| backend/app/analysis/fatigue.py | FatigueTracker class |
| backend/app/analysis/changepoint.py | Changepoint detection (BOCPD/CUSUM) |
| backend/tests/test_fatigue.py | Fatigue tracking tests |

## Tasks

1. Implement rolling statistics engine (5-pitch window)
2. Build fresh baseline computation (first N pitches of outing)
3. Implement changepoint detection (BOCPD via ruptures + CUSUM)
4. Build cross-outing fatigue comparison
5. Generate fatigue report with onset timing and severity

## Dependencies

- Upstream: feature-extraction
- Downstream: api-layer
