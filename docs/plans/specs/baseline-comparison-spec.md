---
name: baseline-comparison
phase: 2
sprint: 2
parent: feature-extraction
depends_on: [feature-extraction]
status: draft
created: 2026-02-16
---

# Baseline Comparison Spec

Establishes what "normal" looks like for a pitcher, then flags deviations.

## Requirements

- Build per-pitcher, per-pitch-type baseline profiles from "good" outings
- Z-score deviation analysis for all biomechanical features
- Flag deviations > 1.5-2.0 standard deviations with severity levels
- Compare new pitches against stored baselines

## Acceptance Criteria

- [ ] Baseline builder: compute mean + std for every feature from N pitches (minimum 20-30 per pitch type)
- [ ] Z-score engine: compute deviation from baseline for any new pitch
- [ ] Severity levels: green (< 1.0 SD), yellow (1.0-1.5 SD), orange (1.5-2.0 SD), red (> 2.0 SD)
- [ ] Per-feature and aggregate deviation scores
- [ ] Baseline stored as PitcherBaseline model, retrievable by pitcher + pitch type
- [ ] Comparison report: list of flagged deviations with magnitude and direction

## Technical Approach

Straightforward Z-score analysis. Baseline = (mean, std) per feature per pitch type. New pitch score = (value - mean) / std. Aggregate score = weighted average of per-feature Z-scores (weight by biomechanical importance — arm slot and release point weighted higher than auxiliary joints).

This is the simplest module but the most broadly useful. Every other module builds on the concept of "deviation from normal."

## Files

| File | Purpose |
|------|---------|
| backend/app/analysis/baseline.py | BaselineBuilder, BaselineComparator |
| backend/tests/test_baseline.py | Baseline tests |

## Tasks

1. Implement BaselineBuilder (aggregate N pitches into mean/std profiles)
2. Implement Z-score deviation engine with severity levels
3. Build comparison report generator
4. Test with synthetic data at known deviations

## Dependencies

- Upstream: feature-extraction
- Downstream: api-layer
