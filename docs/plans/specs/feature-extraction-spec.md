---
name: feature-extraction
phase: 2
sprint: 1
parent: sam3d-inference
depends_on: [sam3d-inference]
status: draft
created: 2026-02-16
updated: 2026-04-03
---

# Feature Extraction Spec

Converts raw joint positions into biomechanical features used by all analysis modules and the mesh-export phase alignment.

## Input Data (from PyTorch/MPS inference)

Two joint sources available:
- `pred_joint_coords (T, 127, 3)` — MHR skeleton joints (FK output, best for biomechanics)
- `pred_keypoints_3d (T, 70, 3)` — MHR70 regressed keypoints (best for visualization)

Joint naming follows MHR70 convention (NOT COCO-17). Key indices for biomechanics:
- Shoulders: 5 (left), 6 (right)
- Elbows: 7 (left), 8 (right)
- Hips: 9 (left), 10 (right)
- Knees: 11 (left), 12 (right)
- Ankles: 13 (left), 14 (right)
- Wrists: 62 (left), 41 (right)
- Full hand keypoints: 21-41 (right), 42-62 (left)

For the 127-joint skeleton, use joint hierarchy from `mhr_body.py` joint_parents array. The 127 joints provide finer-grained body articulation than the 70 keypoints.

## Requirements

- Compute joint angles from 3-joint chains (elbow, shoulder, hip, trunk, knee)
- Compute angular velocities and accelerations (first/second derivatives)
- Detect pitch delivery phases (foot plant, MER, ball release)
- Align pitches to common reference events for comparison
- Compute spatial relationships (hip-shoulder separation, stride length)
- Compute energy flow / kinetic chain timing

## Acceptance Criteria

- [ ] Joint angle mapping: MHR70/MHR127 index → biomechanical joint name (documented, testable)
- [ ] Joint angles computed for all relevant chains: elbow flexion, shoulder abduction, shoulder external rotation, hip flexion, knee flexion, trunk tilt, trunk rotation
- [ ] Angular velocities via Savitzky-Golay or finite differences, with configurable smoothing
- [ ] Phase detection: foot plant (front foot vertical velocity deceleration), MER (peak shoulder external rotation), ball release (hand max forward extension)
- [ ] Pitch alignment to foot plant, MER, or release (configurable)
- [ ] Hip-shoulder separation: angular difference between hip line and shoulder line projected onto horizontal
- [ ] Stride length: normalized by pitcher height (estimated from joint distances)
- [ ] Release point: 3D coordinates of throwing-hand wrist (MHR70 idx 41 for right, 62 for left) at release frame
- [ ] Kinetic chain: time series of pelvis, trunk, shoulder, elbow, wrist peak velocities
- [ ] All features stored in PitchData.joint_angles, .angular_velocities, .phase_boundaries dicts
- [ ] Phase boundaries exported for mesh-export phase alignment

## Technical Approach

Joint angle computation: vectors from joint triplets, angle via `arccos(dot product)`. Use scipy.signal.savgol_filter for smooth derivatives (better than raw finite differences on noisy 3D data). Phase detection via velocity thresholds on specific joints (front foot for foot plant, shoulder for MER, hand for release).

Normative ranges from pitcher-mechanics.md research: validate computed values fall within expected ranges (e.g., MER shoulder external rotation ~170 degrees, hip-shoulder separation 35-60 degrees at peak).

Pitcher handedness detection: compare left vs right wrist velocity during delivery — throwing hand has higher peak velocity. Or accept as input parameter.

## Files

| File | Purpose |
|------|---------|
| backend/app/pipeline/features.py | FeatureExtractor class |
| backend/app/pipeline/joint_map.py | MHR70/MHR127 index → biomechanical name mapping |
| backend/app/pipeline/angles.py | Joint angle computation |
| backend/app/pipeline/phases.py | Pitch phase detection |
| backend/app/pipeline/alignment.py | Pitch-to-pitch alignment |
| backend/app/pipeline/kinetics.py | Kinetic chain / energy flow |
| backend/tests/test_features.py | Feature extraction tests with known values |

## Tasks

1. Build MHR70/MHR127 joint index mapping with named accessors
2. Implement joint angle computation for all relevant chains
3. Implement angular velocity/acceleration with Savitzky-Golay smoothing
4. Build pitch phase detection (foot plant, MER, release)
5. Build pitch alignment engine (align to foot plant / MER / release)
6. Compute spatial features (hip-shoulder sep, stride length, release point)
7. Compute kinetic chain timing sequence
8. Export phase boundaries for mesh-export consumption

## Dependencies

- Upstream: sam3d-inference (provides raw joint data via PyTorch/MPS)
- Downstream: all analysis modules consume extracted features; mesh-export uses phase boundaries for alignment
