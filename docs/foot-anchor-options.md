# Foot Anchor Options — Blender Viewer

Decision doc for how to anchor the pitcher's pivot foot in
`scripts/blender_field_viewer.py` during the NPZ animation playback.

Last updated: 2026-04-07

---

## The problem

SAM 3D Body regresses each frame **independently** and outputs vertices in
body-canonical space (no world translation). When played back in Blender,
the pivot foot (pitcher's back foot) drifts frame-to-frame because:

1. The model's canonical origin wanders slightly per frame (pure model noise).
2. Every pitcher has a unique pre-windup ritual — a small weight-shift, rock,
   or shuffle **before** the leg lift begins — which adds real anatomical
   motion on top of the noise.

We can't tell real motion from model noise using a single foot trajectory,
so a one-size-fits-all anchor either **over-compensates** (body slides
unnaturally to fight real motion) or **under-compensates** (foot drifts off
the rubber from pure noise). A uniform anchor can't model both regimes.

### The two regimes of the pivot foot in real baseball

| Regime | Frames | Pivot foot behavior |
|---|---|---|
| Set position | 0 → leg-lift start | Glued to the rubber, near-zero motion |
| Windup + stride + release + follow-through | leg-lift start → end | Slides, lifts, eventually steps off |

A good anchor should treat these two regimes differently.

---

## Current state (what's implemented today)

**Hybrid anchor** (current default, mode `"ankle"`):

- Static Z offset computed once from frame 0 (lowest vertex → rubber height).
- Per-frame X/Y offset computed to compensate pivot foot lateral drift.
- Auto-detects the pivot foot per clip (whichever ankle has less total motion).

Pros: no vertical bobbing, pivot foot stays near rubber laterally.
Cons: fights real pre-windup motion, body slides laterally by up to 82 cm
  across the clip as the anchor over-corrects.

Also available (mode `"vertex"`): static Z, no X/Y anchor.
Pros: no slide. Cons: foot floats above the mound during the windup.

---

## Options

### Option 1 — Phase-gated anchor

Use the computed phase markers (`set`, `leg lift`, `stride`, `acceleration`,
`release`, `follow-through`) from `_detect_pitch_phases` to switch the anchor
on/off based on phase.

- **Frames `0` → `leg_lift_start` (frame ~105 on the current test clip):**
  anchor pivot foot to the rubber. Set position is held rock-still.
- **Frames `leg_lift_start` → end:** release the anchor. Body and foot move
  freely with whatever the model regressed. Pre-windup step, stride, and
  follow-through all play with their natural motion (including the regression
  warts).

**Pros:** physically correct. Mirrors what a pitcher actually does. Set
position is stable, windup motion is free. You can *see* the transition
between set and motion, which is informative for analysis.

**Cons:** the foot will drift somewhat after the transition because we stop
fighting canonical drift. The transition itself may be visually jarring if
the model's regressed foot position at `leg_lift_start` is far from the
rubber.

**Implementation sketch:** ~10 lines added to the offset computation. For
frames `< leg_lift_start`, use the current hybrid offset. For frames `>=
leg_lift_start`, hold the offset constant at the value from
`leg_lift_start - 1` (so there's no discontinuity; the body just stops being
re-anchored).

---

### Option 2 — Smoothed anchor

Apply a low-pass filter to the per-frame X/Y offset trajectory. The result
is still anchored, but the body slides **gradually** rather than snapping
frame-to-frame. High-frequency model noise is removed; low-frequency real
motion (the pre-windup shuffle) is preserved.

Candidate filters:
- **Savitzky-Golay** (window 11, order 3): good at preserving peaks.
- **Butterworth low-pass** (cutoff ~3 Hz at 60 fps): classic pose smoothing.
- **Simple moving average** (window 7): cheapest, works fine in practice.

**Pros:** the foot stays roughly near the rubber the whole time. Body motion
looks smoother. No phase-boundary discontinuity.

**Cons:** still trading body slide for foot anchor, just less harshly.
Doesn't actually fix the underlying regime mismatch — it just blurs the
artifacts so they're less obvious.

**Implementation sketch:** ~5 lines using `scipy.signal.savgol_filter` or
hand-rolled moving average on `ankle_lock_offsets[:, 0]` and `[:, 1]`.

---

### Option 3 — Drop the anchor entirely

Go back to a single static offset computed from frame 0 only. The body sits
on the rubber at frame 0; everything after is exactly what the model
regressed.

**Pros:** no unnatural body slide. Simplest code. Same behavior as the
static-Z `"vertex"` mode if we delete mode `"ankle"`.

**Cons:** same problem you originally complained about — the pivot foot
drifts off the rubber as the windup progresses. The body floats above the
mound during the leg lift.

**Implementation sketch:** use the existing static Z fallback; no X/Y
compensation at all.

---

### Option 1 + smoothing (phase-gated + low-pass)

Combine Options 1 and 2. Phase-gate the anchor (active during set, released
during motion), AND smooth the anchor trajectory during the active period so
the set position doesn't twitch frame-to-frame.

**Pros:** set position is rock-solid AND smooth. Windup is free and natural.
Transition at phase boundary is gentler because smoothing damps the
last-frame-of-set offset.

**Cons:** more code. Harder to explain. Possibly over-engineered if Option 1
alone works well.

---

## Decision

**To be filled in after testing.**

Leaning toward: **Option 1** first, fall back to **Option 1 + smoothing** if
the set position twitches, fall back to **Option 2** alone if the
phase-boundary transition is too jarring.

---

## Related work (not yet implemented)

### Known limitations not addressed by any option above

1. **No world-space translation**: the body never strides forward through
   the world because the model outputs body-canonical pose only. No option
   above fixes this. To show a real forward stride, we'd need to either
   (a) trust `cam_t` and unproject it into world meters, or (b) integrate
   joint velocity signals to reconstruct a world trajectory.

2. **Follow-through foot-lift**: in reality, the pivot foot eventually lifts
   off the rubber during the late follow-through. None of the options above
   handle that — Options 1 and 2 both keep the anchor near the rubber, and
   Option 3 ignores it entirely. A Phase 2 enhancement could release the
   anchor at `release_frame` (frame ~351) instead of `leg_lift_start`.

3. **Left-handed pitchers**: the pivot foot auto-detector (lower total
   motion) should handle this automatically. Hasn't been tested yet.

4. **Mirrored source video**: if the broadcast feed is mirrored, the pivot
   foot detection might get confused. Unlikely in practice; flag for later.

### Adjacent issues flagged during this session

- Source FPS from the `.npz`: already fixed (capture + stash).
- Axis convention: already fixed via `(-x, -z, -y)` mapping.
- Shape key interpolation: already fixed via user-pref CONSTANT default.
- Detection bbox tracking: already fixed via `_select_bbox`.
- Residual jank in frames 144-193 (the late cluster): NOT fixed. May need
  a Hampel filter on the vertex trajectory (Fix B from earlier discussion).
