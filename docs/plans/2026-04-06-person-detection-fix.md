# Person Detection Reliability — Fix Plan

## Problem

Per-frame person detection in the inference pipeline produces wildly wrong meshes
on **8-21% of frames** because the detector has no temporal continuity. When the
pitcher is occluded, motion-blurred, or partially out of frame, `fasterrcnn`
locks onto whoever else is in the scene (umpire, batter, fans, base coach) and
the body model gets fed the wrong crop.

### Evidence

From `benchmarks/bench_fps.py` results on the Ohtani WS Game 7 clip
(`inn2_ab11_p5_FF_55017c29.mp4`):

| Strategy | Outlier rate | Worst frame jump |
|----------|--------------|------------------|
| 60fps    | 8.9% (16/180)  | 979mm vertex displacement |
| 30fps    | 13.3% (12/90)  | 962mm                     |
| 24fps    | 20.8% (15/72)  | —                         |
| variable | 10.6% (12/113) | —                         |

A pitcher's fastest body part (the hand at release) moves ~25 mm/ms = ~400mm
per 60fps frame. **900mm in one frame is physically impossible** — that's the
detector teleporting to a different person.

### Root cause locations

1. `backend/app/pipeline/inference.py:117-138` (PyTorch backend) — calls
   `detector([img_tensor])` per frame, picks the largest "person" by area, no
   memory of previous frames
2. `backend/app/pipeline/inference.py:187-211` (MLX backend) — calls
   `detect_persons_cached(frame)` and takes `detections[0]`. Same problem.
3. `sam3d_mlx/video.py:51-85` — `track_person()` function exists with IoU
   tracking, but is **only used by the standalone CLI script**, not by the
   backend inference path.

## Goals

- Reduce outlier rate from 8-21% to <2% on the same Ohtani clip
- No regression on clean clips (where detection already works)
- Keep inference time under 1s/frame on M3 Max
- Reusable across PyTorch and MLX backends

## Non-goals

- Fine-tuning the detector on baseball footage (out of scope, weeks of work)
- Multi-person tracking (we only care about the pitcher)
- Real-time/streaming use case (this is offline batch inference)

---

## Phase 1: Stateful tracker (highest leverage, lowest cost)

**Goal**: Wire the existing `track_person()` IoU tracker into the backend
inference path.

### Tasks

1. **Refactor `_BaseInference.process_video_frames`** at
   `backend/app/pipeline/inference.py:28` to maintain a `last_bbox` across
   frame iterations and pass it to `process_frame`.

2. **Update `process_frame` signatures** in both backends to accept an optional
   `prev_bbox: np.ndarray | None` parameter.

3. **Inside MLX `process_frame`**:
   - Get all detections from `detect_persons_cached(frame)` (not just the first)
   - If `prev_bbox` is provided, find the detection with highest IoU vs
     `prev_bbox` (threshold 0.3, matching the existing `track_person`)
   - Fall back to "largest detection" only on the first frame or when IoU fails

4. **Inside PyTorch `process_frame`**: Same logic, applied to the
   `boxes_np` array returned from torchvision.

5. **Return the chosen bbox** so the caller can pass it as `prev_bbox` next
   iteration.

### Files touched

- `backend/app/pipeline/inference.py` (refactor)
- `tests/test_inference.py` (new test for tracking continuity)

### Success metric

Outlier rate drops below 5% on the Ohtani clip. Run the same benchmark to
verify.

### Estimated effort

2-3 hours.

---

## Phase 2: First-frame anchoring

**Goal**: Make the initial bbox selection robust to multi-person scenes.

The IoU tracker only works after frame 0. If the first frame has the umpire
selected instead of the pitcher, every subsequent frame will track the umpire.

### Tasks

1. **Heuristic ranking** for first-frame selection:
   - Prefer detections in the upper-center region of the frame (where the mound
     is in standard broadcast camera angles)
   - Prefer detections with aspect ratio matching a standing/throwing person
     (taller than wide)
   - Penalize detections that overlap with image edges (likely cropped fans)

2. **Optional manual override**: CLI flag `--initial-bbox x,y,w,h` to specify
   the pitcher's bbox manually if the heuristic fails. Useful for batch runs
   where one bad first-frame ruins a whole pitch.

3. **Add metadata field** `initial_bbox_source: "auto" | "manual"` to the
   pitch record so we can audit which clips needed manual intervention.

### Files touched

- `backend/app/pipeline/inference.py` (add `_choose_initial_bbox` helper)
- `scripts/batch_inference.py` (add `--initial-bbox` flag plumbing)
- `backend/app/data/pitch_db.py` (new column on the pitches table)

### Success metric

Manual review of 20 random clips shows 0 wrong-pitcher selections.

### Estimated effort

3-4 hours.

---

## Phase 3: Validation guard

**Goal**: Detect bad bboxes during inference, not after.

Even with tracking, the IoU match can drift. If the bbox suddenly grows 3x or
shifts by 200 pixels, that's almost certainly a tracking failure.

### Tasks

1. **Add bbox sanity checks** between frames:
   - Reject if center moves more than 100px (configurable)
   - Reject if area changes more than 2x
   - Reject if aspect ratio changes more than 50%

2. **On rejection**, fall back to:
   - The previous frame's bbox (assume the pitcher hasn't moved much)
   - Mark the frame as `detection_status: "fallback"` in the result

3. **Track fallback rate** in the pipeline output. High fallback rate = clip
   should be flagged for manual review.

### Files touched

- `backend/app/pipeline/inference.py` (add `_validate_bbox_transition` helper)
- `backend/app/models/pitch.py` (add `detection_status` per frame)

### Success metric

Outlier rate (measured by frame-to-frame vertex displacement > 300mm) drops
below 2%.

### Estimated effort

2 hours.

---

## Phase 4: Re-export benchmark and rerun

**Goal**: Verify the fix works and update the FPS benchmark with clean data.

Once Phases 1-3 land, the FPS benchmark deviation numbers should drop
dramatically because the comparisons won't be poisoned by junk frames anymore.

### Tasks

1. Re-run `python -m benchmarks.bench_fps --clip <ohtani.mp4>` for all four
   strategies
2. Compare results: deviation should drop from ~38mm (30fps) to single-digit mm
3. Re-evaluate which FPS strategy actually wins with clean data
4. Update `docs/plans/findings.md` with the corrected conclusions

### Estimated effort

1 hour (mostly waiting for inference).

---

## Phase 5: Optional — pose-based prediction (if Phases 1-3 aren't enough)

**Goal**: When IoU tracking fails (long occlusion, ball obstructs body), use
the predicted joint positions to forecast where the body should be in the next
frame.

### Approach

1. Take the joint positions from the last 2-3 successful frames
2. Compute velocity per joint (mm/frame)
3. Project forward one frame to get expected joint positions
4. Convert to a bbox prediction and use that as a stronger prior than IoU

This is the gold-standard approach used in pose tracking papers but adds
complexity. **Skip if Phase 3 brings outliers below 2%.**

### Estimated effort

1 day (only if needed).

---

## Out of scope

These were considered and rejected for this plan:

- **ByteTrack / DeepSORT**: Real multi-object trackers. Overkill for
  single-pitcher tracking and add a heavy dependency.
- **Detector fine-tuning**: Fine-tuning fasterrcnn on baseball footage would
  fix the root cause but takes weeks of labeling and training.
- **Switch to YOLO**: Faster detector but same per-frame statelessness.
  Doesn't solve the actual problem.
- **Manual bbox annotation**: Fine for single clips, doesn't scale to a season's
  worth of pitches.

---

## Order of operations

1. **Phase 1** is the biggest win and should land first. Most outliers happen
   when the detector's "first" detection rotates between people; IoU tracking
   alone fixes ~70% of cases.
2. **Phase 4** runs immediately after Phase 1 to measure improvement.
3. **Phase 2** addresses the remaining "first frame was wrong" failures.
4. **Phase 3** catches the tail of weird mid-clip failures.
5. **Phase 5** only if needed.

## Definition of done

- [ ] `track_person` IoU tracking wired into both backend inference paths
- [ ] First-frame heuristic selects the pitcher on 20/20 manually-reviewed clips
- [ ] Bbox validation guard rejects > 99% of teleportation events
- [ ] Re-run FPS benchmark shows < 2% outlier frames at all four FPS strategies
- [ ] Old NPZ files in `data/meshes/` flagged or re-processed with the fix
- [ ] `tests/test_inference.py` covers tracking continuity, fallback behavior,
      and the validation guard
