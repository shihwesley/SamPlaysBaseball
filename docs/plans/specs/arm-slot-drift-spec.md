---
name: arm-slot-drift
phase: 2
sprint: 2
parent: feature-extraction
depends_on: [feature-extraction]
status: draft
created: 2026-02-16
---

# Arm Slot Drift Spec

Tracks gradual or sudden changes in arm angle at release over time.

## Requirements

- Compute arm angle at release for every pitch
- Track within-game drift (pitch-by-pitch)
- Track cross-outing trends (start-to-start)
- Detect bimodal arm slot distributions (tipping risk indicator)
- Distinguish intentional changes from unintentional drift

## Acceptance Criteria

- [ ] Arm slot angle: angle between forearm vector and horizontal plane at release frame
- [ ] Per-pitch tracking with rolling average
- [ ] Within-game drift detection: flag if arm slot changes > 3 degrees from early-outing average
- [ ] Cross-outing trend: plot arm slot by date, detect trend via linear regression
- [ ] Bimodal detection: mixture model test on within-game arm slot distribution
- [ ] Year-over-year comparison: flag 5+ degree changes (from FanGraphs 312-pitcher study: 27% of pitchers)
- [ ] Context flag: new pitch type added? injury history? (manual annotation)
- [ ] Report: current arm slot, trend direction, bimodal risk, comparison to historical

## Technical Approach

Arm angle = arctan2 of forearm vector (elbow → wrist) projected onto sagittal plane. Rolling average (5-pitch window) for within-game. Linear regression for cross-outing trend. Gaussian Mixture Model (n=2) on within-game distribution — if BIC favors 2-component over 1-component, flag as bimodal (potential tipping).

FanGraphs research: 83 of 312 pitchers (27%) changed arm slot by 5+ degrees year-over-year. Bimodal within-game distributions often correlate with pitch type (pitchers dropping arm slot for breaking balls).

## Files

| File | Purpose |
|------|---------|
| backend/app/analysis/arm_slot.py | ArmSlotTracker class |
| backend/tests/test_arm_slot.py | Arm slot tests |

## Tasks

1. Implement arm slot angle computation at release
2. Build within-game drift detection
3. Build cross-outing trend analysis (linear regression)
4. Implement bimodal distribution detection (GMM)
5. Generate arm slot report with trend + risk flags

## Dependencies

- Upstream: feature-extraction
- Downstream: api-layer
