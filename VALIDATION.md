# VALIDATION

What this tool can and cannot tell you about a pitcher's mechanics.

This document exists because a pitching coach's first question is always *"how do I know these numbers are right?"* — and the honest answer matters more than the impressive feature list. Read this before trusting any specific number the dashboard produces.

---

## TL;DR

| Aspect | Trustworthy? |
|--------|--------------|
| **Pairwise delivery comparison** (same pitcher, same camera, same outing) | **High** |
| **Phase detection and timing** (windup, leg lift, foot strike, MER, release) | **High** |
| **In-plane joint angles** (hip flex, knee flex, trunk lean — viewed from the side) | **Medium-high** |
| **Tipping confirmation** (post-game, in-plane body posture differences between pitch types) | **Medium-high** |
| **3D mesh visualization** (the visual artifact) | **High** |
| **Absolute joint positions** (release point xyz, arm slot angle in world coordinates) | **Medium** — useful for trends, suspect for absolute claims |
| **Hip-shoulder separation in degrees** | **Medium** — depth-axis ambiguity affects this directly |
| **Velocity and spin** | **Not measured** — use Statcast |
| **Pitch trajectory and location** | **Not measured** — use Hawk-Eye |
| **Elbow torque, UCL stress, injury prediction** | **Not measured** — see Future Biomechanics Work |

---

## The cornerstone number: 54.8mm MPJPE

SAM 3D Body achieves **54.8mm Mean Per Joint Position Error** on standard 3D pose benchmarks. KinaTrax, the marker-based system every MLB team uses, achieves **~25mm** with eight synchronized cameras permanently installed in the stadium.

That's roughly **2x the error of professional motion capture**, from a single broadcast camera, with no hardware install. That tradeoff is the entire premise of this project. It's good enough for the **relative** comparisons that drive most player-development decisions ("did this delivery look different from his last delivery?"). It's not good enough for the **absolute** claims that sports medicine relies on ("his elbow torque is 64 Nm").

Every claim in this tool should be read through that lens.

---

## What this tool measures *well*

### 1. Pairwise delivery comparison (high confidence)

`compare_deliveries()` is the most trustworthy thing in the codebase. Here's why: when you compare two deliveries from the **same pitcher**, **same camera angle**, **same outing**, the systematic errors of the model **cancel out**. If the model is biased toward over-estimating arm slot by 4° on this camera, it will over-estimate it by 4° on both deliveries, and the *difference* will still be valid.

This is the core insight. Relative measurements are reliable even when absolute measurements are noisy.

What this means in practice:
- "First-inning fastballs vs sixth-inning fastballs" → trust it
- "Today's slider release vs his slider release in yesterday's outing, same camera" → trust it with caveats (camera may have moved between games)
- "His arm slot is 78°" → don't trust the number. Trust the *change* in the number across deliveries.

### 2. Phase detection and timing (high confidence)

Phase boundaries (knee lift, foot strike, max external rotation, release) are detected from joint velocity signals, which are **derivatives of position over time**. Even when individual positions are noisy, derivatives smoothed across 5-10 frames are stable. The pipeline's phase detection has been sanity-checked against video frame-by-frame on Ohtani and matches visual inspection.

### 3. In-plane joint angles (medium-high confidence)

When the camera is roughly perpendicular to the throwing plane (standard center-field or first-base broadcast cam), the following measurements are in the camera plane and depth ambiguity does **not** corrupt them significantly:

- Knee flexion at foot strike (front and back leg)
- Trunk lateral lean
- Hip flexion
- Stride length (when measured in pixels normalized to height)
- Stride direction (closed/open) — within camera-plane resolution

### 4. Tipping confirmation (medium-high confidence — see "What this is for" below)

This tool's tipping module is **not a tipping predictor**. It is a **post-game confirmation tool**. The use case is:

> A coach or opposing-team scout sees something live and says "I think he's tipping his slider." After the game, you run the analysis tool against that outing, grouped by pitch type, and the tool surfaces the *measurable* in-plane body-posture differences between his slider deliveries and his fastball deliveries — glove pre-set position, head/shoulder posture, hand placement, pre-delivery rhythm differences. You then have something concrete to show the player.

This works because:
- Tipping tells are typically **in-plane body posture differences** (glove height, shoulder set, where the hand sits before going into the windup) — not depth-axis claims
- The "label" is provided by the human observation, not learned from a dataset
- The tool's job is to *quantify and locate* what the coach already saw, frame by frame

What the tool will **not** do:
- Predict tipping from scratch on an unlabeled pitcher
- Find tipping the coaches missed (probably — though it may surface candidates worth investigating)
- See into the glove or detect grip tells (the Astros' 2017 method) — the standard broadcast cam doesn't see the grip

---

## What this tool measures *with caveats*

### Arm slot angle (medium confidence)

Arm slot is calculated from the 3D position of the shoulder, elbow, and wrist at release. On a perpendicular pitcher cam, the shoulder-elbow-wrist plane is partially in the camera plane and partially in the depth axis. Single-camera depth estimation has the largest error precisely along this axis.

**Use it for:** Detecting *changes* in arm slot across deliveries (Δ within an outing, or Δ between outings on the same camera).
**Don't use it for:** Reporting an absolute arm-slot angle to a coach as a definitive number.

Estimated error: ±5° on absolute angles, ±2° on within-outing deltas (estimate from monocular pose literature; not yet validated against marker mocap).

### Release point (medium confidence)

Release point xyz comes from wrist position at the release frame. Same depth-axis problem: x and y are reliable in the camera plane, z (depth) has the largest error.

**Use it for:** Detecting release-window *consistency* — is the pitcher hitting the same release window across pitches of the same type?
**Don't use it for:** Comparing release-point depth in absolute world coordinates to a different pitcher or a different camera.

Estimated error: ±3 cm in-plane, ±10 cm in depth (literature estimate).

### Hip-shoulder separation in degrees (medium confidence)

Hip-shoulder separation is the angle between the pelvis-vector and the shoulder-vector at peak. Both vectors involve depth, both involve angular geometry, the product compounds the error.

**Use it for:** Trend comparisons within the same camera setup.
**Don't use it for:** Comparison to published norms (35-60°) without explicit caveats.

---

## What this tool does *not* measure

These are listed because the absence of them matters more than the presence of features that overpromise.

- **Velocity** — use Statcast. The tool ingests Statcast velocity for context but does not derive it from video.
- **Spin rate** — use Statcast.
- **Pitch trajectory and location** — use Hawk-Eye.
- **Elbow torque / shoulder torque** — requires inverse dynamics with mass distribution and force plates. Not derivable from kinematics alone.
- **UCL stress / injury prediction** — requires the above plus longitudinal cohort data. The injury_risk module that previously existed has been moved to the Future Biomechanics Work section because the inputs it relies on are not yet trustworthy enough to support medical-adjacent claims.
- **Fatigue prediction** — same. Currently in Future Biomechanics Work.
- **Pitch outcome correlation** — this requires reliable per-pitch matching between video and Statcast records, which works when using pre-segmented Baseball Savant clips, but is unsolved for raw broadcast footage.

---

## How the numbers were validated

### What HAS been validated

| Component | Validation | Result |
|-----------|------------|--------|
| MLX SAM 3D Body port vs PyTorch reference | Vertex-level numerical comparison across 18,439 vertices on 50+ frames | **<0.001mm error** — full numerical parity |
| Phase detection | Manual frame-by-frame inspection on Ohtani 2024 outings | Matches visual inspection within 1-2 frames |
| MLX inference speed | Benchmarked vs PyTorch/MPS on M3 Max | 488ms vs 667ms median, 1.37x faster |
| Backend correctness | 263 backend tests passing (pytest) | Green |
| Frontend type safety | tsc --noEmit | 0 type errors |

### What has NOT been validated (but we know how)

| Component | Validation pathway | Status |
|-----------|--------------------|--------|
| Joint-angle accuracy vs marker mocap | Compare pipeline output against [Driveline OpenBiomechanics dataset](https://github.com/drivelineresearch/openbiomechanics) (100K+ marker-mocap pitches) | Identified, not yet run |
| Arm-slot error bars vs ground truth | Same | Identified, not yet run |
| Release-point depth error | Same | Identified, not yet run |
| Generalization across camera angles | Test on multiple broadcast feeds with known calibration | Not yet run |

The validation path is clear. It hasn't been run yet because the demo storytelling work was prioritized first. That's an honest tradeoff, not a hidden gap.

---

## Future Biomechanics Work

These modules exist in the codebase but have been **moved out of the user-facing flow** because they require ground truth or data this project does not yet have. The code is preserved (see `backend/app/analysis/`) so the work can be revived when the validation backing exists.

### `injury_risk.py`
**What it tried to do:** Composite "injury risk indicator" combining fatigue, arm slot drift, and timing anomalies into a traffic-light score.
**Why it's hidden:** Combining three unvalidated signals into a medical-adjacent score is exactly the failure mode that makes team doctors reject biomechanics tools. Even with the "indicator not predictor" honest framing, the inputs aren't trustworthy enough yet.
**Path to revival:** Validate each input signal against marker mocap on the Driveline OpenBiomechanics dataset, then validate the composite against published UCL injury cohort data.

### `fatigue.py`
**What it tried to do:** Bayesian online changepoint detection on within-game pitch sequences to find sudden mechanical breakdowns.
**Why it's hidden:** Changepoint detection on 5-20 broadcast clips per game produces too many false positives without a confidence-interval story. Needs dense within-game pitch coverage and longitudinal baselines from the same pitcher.
**Path to revival:** Pair with high-frequency in-game pitch capture (TrackMan-linked) and validate against pitch-velocity decline as a proxy ground truth.

### `command.py`
**What it tried to do:** Release-point clustering correlated with pitch location and miss patterns.
**Why it's hidden:** Requires pitch trajectory ground truth (where the ball ended up) which means pairing with Hawk-Eye data. The input side is unsolved.
**Path to revival:** Pair with Hawk-Eye trajectory data per pitch, then re-validate the release-cluster-to-location mapping.

### `arm_slot_drift.py` (kept, but with caveats)
**Current status:** Still in the user-facing flow because release-window *consistency within an outing* is a useful signal even with depth error. The "27% of pitchers shift 5+ degrees year-over-year" claim from the README is from population-level research and is **not** something this tool can confirm on a single pitcher with limited data — it's context, not a derived result.

---

## Honest things a coach or analyst should know before using this

1. **The first time you query a new pitcher, expect ~1 minute per pitch of inference time.** Cached after that.
2. **The tool compares deliveries from the same camera angle.** Don't expect meaningful comparisons across different broadcast cameras without manual calibration.
3. **All "trend" claims require enough data to support them.** A drift detected across 5 pitches is not the same as a drift across 50.
4. **The 3D mesh is beautiful but the joint numbers driving it inherit single-camera depth ambiguity.** Pretty visualization ≠ ground-truth biomechanics.
5. **No medical claims.** Nothing in this tool is medical advice. The injury_risk module has been removed from the flow specifically to avoid that framing.

---

## What this tool *is* for

A post-game review tool for analysts and coaches who want to:

- **Confirm what the field saw** — coaches and players notice things during games; this tool quantifies and locates the mechanical evidence frame by frame
- **Compare deliveries side by side** in 3D — first inning vs sixth inning, slider vs fastball, today vs last week
- **Render shareable analysis clips** with mesh ghost overlays and joint highlights
- **Generate scouting-language narrative reports** from the structured comparison data
- **Run all of this locally on Apple Silicon** with no cloud cost and no API dependency

It is **not** a replacement for KinaTrax, Hawk-Eye Pose, or marker-based mocap. It is a **complement** for situations where those systems are unavailable: bullpen sessions, archived broadcast footage, scouting tape, college and amateur footage, post-game film review on a laptop.

---

*Last updated: 2026-04-07*
