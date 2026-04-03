---
name: tipping-detection
phase: 2
sprint: 2
parent: feature-extraction
depends_on: [feature-extraction]
status: draft
created: 2026-02-16
---

# Tipping Detection Spec

Detects when a pitcher reveals upcoming pitch type through consistent mechanical differences visible before the pitch arrives at the plate.

## Requirements

- Extract pre-release features (wind-up through arm cocking phases only)
- Train pitch-type classifier on pre-release mechanics
- If cross-validated accuracy exceeds chance, tipping exists
- Identify which features drive the classification (feature importance + SHAP)
- Report specific mechanical tells with frame-level timing

## Acceptance Criteria

- [ ] Pre-release feature extraction: features from wind-up, stride, and arm cocking phases only (nothing after MER)
- [ ] Classifier: Random Forest or XGBoost, 5-fold cross-validation
- [ ] Tipping threshold: accuracy > chance level (25% for 4-pitch mix, 33% for 3, 50% for 2)
- [ ] Feature importance via permutation importance + SHAP values
- [ ] Direction of difference: "glove 2.3cm higher on changeups" not just "glove position matters"
- [ ] Frame-level annotation: which frames in the delivery show the tell
- [ ] Minimum sample requirement: 50+ pitches with type labels
- [ ] Report: per-feature breakdown with visual comparison across pitch types

## Technical Approach

From pitcher-mechanics.md: cluster pitches by type, extract pre-release feature vectors, train classifier. Key pre-release features: glove position/height in set, head tilt, shoulder asymmetry in wind-up, trunk angle at stride start, arm path during cocking, phase timing differences.

Use scikit-learn RandomForest + XGBoost. SHAP TreeExplainer for interpretable feature contributions. Malter Analytics showed this works with just 25 OpenPose keypoints — 127 MHR joints should pick up subtler tells.

## Files

| File | Purpose |
|------|---------|
| backend/app/analysis/tipping.py | TippingDetector class |
| backend/app/analysis/shap_utils.py | SHAP wrapper for analysis modules |
| backend/tests/test_tipping.py | Tipping detection tests |

## Tasks

1. Implement pre-release feature extraction (wind-up through cocking only)
2. Build classifier training pipeline (RF + XGBoost, cross-validation)
3. Implement SHAP feature importance analysis
4. Build tipping report generator with per-feature directional breakdown
5. Test with synthetic data (known pitch-type differences injected)

## Dependencies

- Upstream: feature-extraction
- Downstream: api-layer
