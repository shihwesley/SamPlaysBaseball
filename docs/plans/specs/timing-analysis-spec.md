---
name: timing-analysis
phase: 2
sprint: 2
parent: feature-extraction
depends_on: [feature-extraction]
status: draft
created: 2026-02-16
---

# Timing Analysis Spec

Analyzes the kinetic chain sequence — the order and timing of energy transfer from ground through ball.

## Requirements

- Extract peak angular velocity timing for each body segment in the kinetic chain
- Verify correct sequencing: pelvis → trunk → shoulder → elbow → wrist
- Detect timing irregularities ("all arm" delivery, simultaneous hip-trunk rotation)
- Quantify hip-shoulder separation dynamics
- Flag energy leaks and compensation patterns

## Acceptance Criteria

- [ ] Peak velocity extraction: timing of max angular velocity for pelvis, trunk, shoulder, elbow, wrist
- [ ] Sequence validation: peaks should occur in proximal-to-distal order
- [ ] Timing gaps: ms between each sequential peak (normative: ~9.5ms per segment for velocity)
- [ ] Hip-shoulder separation: peak angle (normative: 35-60 degrees) + timing relative to foot plant
- [ ] "All arm" detection: trunk/hip contribution below threshold
- [ ] Simultaneous rotation detection: hip-trunk timing gap < 5ms
- [ ] Late arm detection: arm peak after trunk peak by > expected delay
- [ ] Energy contribution: decompose each joint's motion into pitch-direction vs lateral vs vertical
- [ ] Wasted motion index: high total motion but low forward-direction contribution per joint
- [ ] Report: kinetic chain sequence, timing gaps, deviation from optimal, energy efficiency

## Technical Approach

Angular velocity peaks from feature-extraction's derivatives. Sequence check is simple ordering validation. Hip-shoulder separation is the angular difference between hip line and shoulder line projected onto horizontal plane — track over time, find peak, compare to normative 35-60 degrees.

Energy decomposition: project each joint's velocity vector onto the pitch-direction axis (roughly from rubber to plate). Ratio of forward-component energy to total energy = efficiency score. Low ratio = wasted motion. Flagged joints = high total motion, low forward contribution.

Normative values from pitcher-mechanics.md: pelvis peaks at ~655 deg/s, hip-shoulder sep 35-60 degrees, MER-to-release ~30-50ms.

## Files

| File | Purpose |
|------|---------|
| backend/app/analysis/timing.py | TimingAnalyzer class |
| backend/app/analysis/energy.py | Energy decomposition, wasted motion detection |
| backend/tests/test_timing.py | Timing analysis tests |

## Tasks

1. Implement peak angular velocity extraction per body segment
2. Build kinetic chain sequence validator
3. Implement hip-shoulder separation dynamics tracking
4. Build energy decomposition and wasted motion index
5. Generate timing report with sequence, gaps, and efficiency scores

## Dependencies

- Upstream: feature-extraction
- Downstream: api-layer
