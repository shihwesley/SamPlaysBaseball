# Session State — 2026-04-07 Trim Pipeline Work

**Status at end of session:** trim pipeline built and partially validated. Final
re-test with refreshed thresholds was not run before context-clear. Pick up here.

**How to resume:**
1. Open a new conversation
2. Run `/workin samplaysbaseball`
3. Say: "Read `docs/plans/2026-04-07-trim-pipeline-state.md` and pick up at TODO-RESUME-001"

There is no `/plan-ceo-review -resume` command. The closest purpose-built tool is
`/checkpoint`, but the doc you're reading right now is the manual checkpoint and
should be the source of truth.

---

## Where the work is right now

| File | Status | What's in it |
|------|--------|--------------|
| `sam3d_mlx/sam31_detector.py` | **shipped, validated** | SAM 3.1 wrapper. `Sam31PitcherDetector` class with calibrated geometric tiebreaker (lower-half + standing + center-frame). Module-level cached singleton via `get_pitcher_detector()`. |
| `backend/app/pipeline/delivery_window.py` | **shipped, NOT yet re-tested** | Refactored at end of session. Two-phase: SAM 3.1 early-exit scan for set frame + cv2 histogram diff for broadcast cut. Tunables `_MIN_CONSECUTIVE_DETECTIONS=1` and `_HIST_CORR_CUT_THRESHOLD=0.85` were just edited but the test was not re-run. **First task in next session is to re-run this test.** |
| `scripts/test_sam31_on_darvish.py` | shipped | One-shot SAM 3.1 sanity check on a single frame. Already passed. |
| `scripts/test_delivery_window.py` | shipped | End-to-end trim test runner. Used to validate both clips. |
| `scripts/fetch_savant_clips.py` | **NOT YET TOUCHED** | The trimmer is built but not wired into the download flow. Task #23 / TODO-RESUME-002. |
| `backend/requirements.txt` | **NOT YET UPDATED** | mlx-vlm>=0.4.4 needs to be added. Task #19 / TODO-RESUME-004. |
| `backend/tests/test_delivery_window.py` | **NOT WRITTEN** | Unit tests pending. Task #15 / TODO-RESUME-005. |

---

## In-flight task list (what was open in TaskCreate when context cleared)

```
#10  ✅ Research: scene detection + person detection options in existing code
#11  ✅ Design delivery_window detector algorithm
#12  ✅ REWRITE: delivery_window.py for SAM 3.1 set-frame detection
#13  ✅ Wire --auto-trim flag (SUPERSEDED — trimming moved to download step)
#14  ✅ Test detector on both Darvish CH clips
#15  ⏳ PENDING — Add unit tests for delivery_window detector
#16  ✅ Decide SAM 3.1 speed strategy with real numbers
#17  ✅ Adapt delivery_window.py to use SAM 3.1
#18  ✅ Create sam3d_mlx/sam31_detector.py wrapper module
#19  ⏳ PENDING — Add mlx-vlm to backend/requirements.txt
#20  ✅ Write trimmed .mp4 output (cv2 VideoWriter mp4v)
#21  ⏳ PENDING — Test full trim+inference pipeline on Darvish CH clip 2 (Ball)
#22  ⏳ IN-FLIGHT — Refactor delivery_window.py: early-exit SAM scan + cv2 scene cut
       (code refactor done, threshold-fix re-test NOT done)
#23  ⏳ PENDING — Integrate trimmer into fetch_savant_clips.py
```

---

## The exact next step (TODO-RESUME-001)

Run this:

```bash
cd /Users/quartershots/Source/SamPlaysBaseball
python3 scripts/test_delivery_window.py
```

**Expected output for the in-play CH clip (b69a4fbd):**
```
detection time: ~5s          (was 11.7s before refactor, was 65.6s before that)
result:
  start_frame: 0  or  ~6      (depends on whether set_frame=6 with min_consecutive=1)
  end_frame:   155            (driven by scene cut detection at frame 155)
  total_frames: 300
  fps: 29.97
  set_frame: 6 (or similar early frame)
  scene_cut_frame: 155        (THIS IS THE KEY VALUE — was None before threshold fix)
  n_frames_kept: ~150
  duration_kept_s: ~5.0
  n_sam_calls: 2-3            (early-exit scan)
```

**If `scene_cut_frame: None` again:** the `_HIST_CORR_CUT_THRESHOLD=0.85` change didn't take effect. Check the file. The previous test run with threshold 0.55 returned `scene_cut_frame: None` and trim window `[15, 180)` which was wrong.

**Then run on the Ball CH clip:**
```bash
python3 scripts/test_delivery_window.py data/clips/526517/inn1_ab5_p1_CH_dd857f31.mp4
```

**Expected:**
```
set_frame: 0
scene_cut_frame: None        (no broadcast cut in a "Ball" outcome clip)
end_frame: 150                (biomechanical cap)
n_sam_calls: ~2
```

If both clips produce the expected results, the trim pipeline is fully validated and you can proceed to TODO-RESUME-002 (integrate into fetch_savant_clips.py).

---

## Calibrated tunables (values + rationale)

Both files have inline comments explaining the calibration. Quick reference:

### `sam3d_mlx/sam31_detector.py` — pitcher tiebreaker
```python
_MIN_ASPECT_RATIO              = 1.5    # tall vs square
_MIN_BBOX_HEIGHT_FRACTION      = 0.35   # excludes catcher, umpire, batter, fielders
_MAX_BBOX_HEIGHT_FRACTION      = 0.85   # excludes glove/face closeups
_LOWER_HALF_THRESHOLD          = 0.55   # excludes batter (~0.43), umpire (~0.43)
_CENTER_HALF_THRESHOLD         = 0.20   # excludes dugout/fans at edges
```
Calibrated against `data/clips/526517/_frame90_HOME.jpg` (Darvish 2017 WS G7
center-field broadcast cam). Loosening any of these will let in batter/catcher
false positives. Tightening risks excluding the pitcher in clips where the
camera is further away.

### `backend/app/pipeline/delivery_window.py` — trim pipeline
```python
_SAMPLE_INTERVAL_SECONDS       = 0.2    # FPS-aware: 6fr@30fps, 12fr@60fps
_MIN_CONSECUTIVE_DETECTIONS    = 1      # JUST CHANGED FROM 2 — verify in next test
_MAX_SCAN_SECONDS              = 8.0    # max scan budget for set-frame search
_HIST_CORR_CUT_THRESHOLD       = 0.85   # JUST CHANGED FROM 0.55 — verify in next test
_PRE_SET_BUFFER_SECONDS        = 0.5    # pre-set keep margin
_POST_SET_DELIVERY_SECONDS     = 5.0    # biomechanical max
_POST_SCENE_CUT_BUFFER_SECONDS = 0.0    # don't extend past a cut
```

The two `JUST CHANGED` values are why the test must be re-run before proceeding.

---

## Demo target — locked

| What | Value |
|------|-------|
| Pitcher | Yu Darvish (MLBAM 506433) |
| Demo game | 2017 WS Game 7 — `game_pk 526517` |
| Demo opponent | Houston Astros (Astros allegedly picked up a slider tip) |
| Demo pitch counts | FF=13, ST=17, FC=5, CU=7, SI=3, CH=2 (47 total) |
| Control game | 2017-09-19 vs Phillies — `game_pk 492355` |
| Control pitch counts | FF=21, ST=21, FC=30, SI=17, CU=6, CH=2 (97 total, no allegations) |
| Working clip angle | `--angle HOME` (NOT the script default of `--angle AWAY` which fails for 2017) |
| Test clips on disk | `data/clips/526517/inn1_ab4_p1_CH_b69a4fbd.mp4` (in-play, broadcast cut at 155) |
|                    | `data/clips/526517/inn1_ab5_p1_CH_dd857f31.mp4` (Ball, no cut) |

---

## Speed budget — measured numbers, not estimates

| Operation | Cost on M3 Max |
|-----------|----------------|
| SAM 3.1 first download | ~48s (one-time) |
| SAM 3.1 cold load | ~5s |
| SAM 3.1 single inference (1008px) | ~1000ms |
| SAM 3.1 first call (JIT warmup) | ~4700ms |
| cv2 histogram correlation per frame | ~0.5ms |
| cv2 scene cut scan over 150 frames | ~0.2s |
| Trim pipeline per clip (refactored) | **~3-5s** (target — to be re-confirmed in next test) |
| Trim pipeline per clip (uniform sampling, deprecated) | ~60s |
| SAM 3D Body inference per frame (MLX) | ~500ms |
| Full pitch inference (300 frames) | ~150s |
| Full pitch inference (150 frames trimmed) | ~75s |

Projected for the demo data:
- Trim 47 WS G7 clips: ~3-4 minutes
- Trim 97 control clips: ~6-8 minutes
- Inference 47 trimmed WS G7 clips: ~60 minutes
- Inference 97 trimmed control clips: ~120 minutes

Total demo prep: roughly **3 hours unattended overnight batch.**

---

## What's been shipped this session that's NOT in flight

These are stable, validated, won't change in the next session unless you ask:

- `VALIDATION.md` (~330 lines) — what's trustworthy, what isn't, with literature error bars
- `README.md` rewritten — product story, dropped MLB-only audience, surfaced deferred modules honestly
- `tipping_detection` module repositioned + new `compare_within_outing()` post-game entry point
- `injury_risk`, `fatigue_tracking`, `command_analysis` unwired from API + reports (.py preserved)
- `historical_legends` spec deleted; manifest cleaned
- Strategic-pivot session note added to `progress.md` (morning)
- Trim-pipeline session note added to `progress.md` (afternoon — this session)
- Pitcher tiebreaker thresholds calibrated and documented
- Both Darvish CH clips downloaded, manifested, ingested into PitchDB
- One Darvish CH pitch (b69a4fbd) fully inferenced through SAM 3D Body — `data/meshes/526517/b69a4fbd.npz` (60 MB) sitting on disk, ready to render
- One trimmed clip on disk: `data/clips/526517/inn1_ab4_p1_CH_b69a4fbd_trimmed.mp4` (from the previous algorithm — re-trim it after the new test)

---

## Git state at session end

Many uncommitted changes. Pre-existing modifications (from before this session) plus all the new files written today. **Nothing has been committed yet.** Suggested next-session commit boundaries:

1. **Strategic pivot batch** (morning work):
   - VALIDATION.md, README.md, tipping.py reposition, injury_risk/fatigue/command unwire, historical_legends delete, manifest/progress/TODOS updates from morning
2. **Trim pipeline batch** (afternoon work):
   - sam31_detector.py, delivery_window.py, test scripts, requirements.txt update (TODO-RESUME-004), this state file
3. **Test fixtures** (when committing the unit tests):
   - The two Darvish CH MP4s + the trimmed outputs as test fixtures

The pre-existing GLB test failures (test_glb_export.py) are from uncommitted work that predates this session — flag in next session but don't try to fix without context.
