---
name: command-analysis
phase: 2
sprint: 2
parent: feature-extraction
depends_on: [feature-extraction]
status: draft
created: 2026-02-16
---

# Command Analysis Spec

Identifies mechanical reasons why a pitcher can't locate pitches where intended.

## Requirements

- Map release point scatter per pitch type (3D centroid + standard deviation)
- Correlate release point variance with mechanical causes
- Identify which mechanical flaws predict missed locations
- Report actionable mechanical fixes

## Acceptance Criteria

- [ ] Release point scatter: 3D centroid and SD per pitch type
- [ ] Confidence ellipse: 2D projection for visualization
- [ ] Correlation analysis: release point variance vs trunk rotation timing, glove arm position, landing foot direction, hip-shoulder separation at release
- [ ] Mechanical predictor ranking: which flaw explains the most variance in miss direction
- [ ] Per-pitch-type breakdown: FB might be tight while SL is scattered
- [ ] Actionable output: "when trunk opens 15ms early, pitches miss arm-side by X inches"
- [ ] Minimum data: 100+ pitches with location data for correlation analysis
- [ ] Works without location data: still reports release point consistency and mechanical correlates

## Technical Approach

Release point from hand joint at release frame. Scatter via standard deviation of 3D coordinates. Correlation via linear regression / GAM (generalized additive model) between mechanical features and release point position. If pitch location data available (from Statcast or manual charting), correlate mechanics → miss direction.

Key mechanical causes from pitcher-mechanics.md: early trunk rotation, front-side pull, landing foot misdirection, hip mobility restriction, anterior core weakness.

## Files

| File | Purpose |
|------|---------|
| backend/app/analysis/command.py | CommandAnalyzer class |
| backend/tests/test_command.py | Command analysis tests |

## Tasks

1. Implement release point scatter computation (3D centroid + SD + confidence ellipse)
2. Build mechanical correlation engine (release variance vs mechanics)
3. Implement miss direction predictor (if location data available)
4. Generate command report with actionable mechanical fixes

## Dependencies

- Upstream: feature-extraction
- Downstream: api-layer
