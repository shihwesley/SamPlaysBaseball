---
name: feature-extraction
phase: 2
sprint: 1
parent: sam3d-inference
depends_on: [sam3d-inference]
status: draft
created: 2026-02-16
---

# Feature Extraction Spec

Converts raw `(T, 127, 3)` joint positions into biomechanical features used by all 6 analysis modules.

## Requirements

- Compute joint angles from 3-joint chains (elbow, shoulder, hip, trunk, knee)
- Compute angular velocities and accelerations (first/second derivatives)
- Detect pitch delivery phases (foot plant, MER, ball release)
- Align pitches to common reference events for comparison
- Compute spatial relationships (hip-shoulder separation, stride length)
- Compute energy flow / kinetic chain timing

## Acceptance Criteria

- [ ] Joint angles computed for all relevant chains: elbow flexion, shoulder abduction, shoulder external rotation, hip flexion, knee flexion, trunk tilt, trunk rotation
- [ ] Angular velocities via Savitzky-Golay or finite differences, with configurable smoothing
- [ ] Phase detection: foot plant (front foot vertical velocity deceleration), MER (peak shoulder external rotation), ball release (hand max forward extension)
- [ ] Pitch alignment to foot plant, MER, or release (configurable)
- [ ] Hip-shoulder separation: angular difference between hip line and shoulder line projected onto horizontal
- [ ] Stride length: normalized by pitcher height
- [ ] Release point: 3D coordinates of dominant hand at release frame
- [ ] Kinetic chain: time series of pelvis, trunk, shoulder, elbow, wrist peak velocities
- [ ] All features stored in PitchData.joint_angles, .angular_velocities, .phase_boundaries dicts

## Technical Approach

Joint angle computation: vectors from joint triplets, angle via `arccos(dot product)`. Use scipy.signal.savgol_filter for smooth derivatives (better than raw finite differences on noisy 3D data). Phase detection via velocity thresholds on specific joints (front foot for foot plant, shoulder for MER, hand for release).

Normative ranges from pitcher-mechanics.md research: validate computed values fall within expected ranges (e.g., MER shoulder external rotation ~170 degrees, hip-shoulder separation 35-60 degrees at peak).

## Files

| File | Purpose |
|------|---------|
| backend/app/pipeline/features.py | FeatureExtractor class |
| backend/app/pipeline/angles.py | Joint angle computation |
| backend/app/pipeline/phases.py | Pitch phase detection |
| backend/app/pipeline/alignment.py | Pitch-to-pitch alignment |
| backend/app/pipeline/kinetics.py | Kinetic chain / energy flow |
| backend/tests/test_features.py | Feature extraction tests with known values |

## Tasks

1. Implement joint angle computation for all relevant chains
2. Implement angular velocity/acceleration with Savitzky-Golay smoothing
3. Build pitch phase detection (foot plant, MER, release)
4. Build pitch alignment engine (align to foot plant / MER / release)
5. Compute spatial features (hip-shoulder sep, stride length, release point)
6. Compute kinetic chain timing sequence

## Dependencies

- Upstream: sam3d-inference (provides raw joint data)
- Downstream: all 6 analysis modules consume extracted features
