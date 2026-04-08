# TODOS

## P0 — RESUME HERE NEXT SESSION (trim pipeline in flight)

These are tasks left in flight at the end of the 2026-04-07 afternoon session. The trim pipeline is built but the final validation pass is not done. **Read `docs/plans/2026-04-07-trim-pipeline-state.md` first** for the full session-state context.

### TODO-RESUME-001: Re-test refactored delivery_window with fresh thresholds
- **What:** Run `python scripts/test_delivery_window.py` on `data/clips/526517/inn1_ab4_p1_CH_b69a4fbd.mp4`. Expected output: `set_frame=6, scene_cut_frame=155, trim window [0, 155)`. Then run on `data/clips/526517/inn1_ab5_p1_CH_dd857f31.mp4`. Expected: `set_frame=0, scene_cut_frame=None, trim window [0, 150)`.
- **Why:** The refactor (`min_consecutive=1`, `_HIST_CORR_CUT_THRESHOLD=0.85`) was applied at the end of the session but never re-tested. This is the gating step before any other trim work proceeds.
- **Effort:** S (10 minutes — two test runs at ~5s each)
- **Files:** `backend/app/pipeline/delivery_window.py`, `sam3d_mlx/sam31_detector.py` (both have edited tunables)
- **If it fails:** check the in-flight task list in `docs/plans/2026-04-07-trim-pipeline-state.md`

### TODO-RESUME-002: Integrate trimmer into fetch_savant_clips.py (task #23)
- **What:** After each successful clip download in `scripts/fetch_savant_clips.py:fetch_game_clips()`, call `find_delivery_window()` and `write_trimmed_clip()`. Update the manifest to record both `raw_video_path` (original) and `video_path` (trimmed). The DB stores the trimmed path. No `--auto-trim` flag — trimming is unconditional.
- **Why:** The trimmer is built but not wired into the download flow. Currently you have to call it manually via `scripts/test_delivery_window.py`. Once wired, every new clip downloads pre-trimmed.
- **Effort:** S (~30 min)
- **Depends on:** TODO-RESUME-001 passes

### TODO-RESUME-003: End-to-end SAM 3D Body inference on a trimmed clip (task #21)
- **What:** Run `python scripts/batch_inference.py --play-id dd857f31-824e-49cf-a17d-5f3f1ebefa7a` against the **trimmed** Ball CH clip. Then run `python -m sam3d_mlx.video --input <trimmed.mp4> --output _overlay_trimmed.mp4` to generate the overlay video. Verify visually that the model tracks Darvish through the entire trimmed clip without losing him.
- **Why:** Visual frame checks proved the trim window is correct, but no SAM 3D Body inference has been run on a trimmed file end-to-end. Subtle codec/fps/index issues could lurk.
- **Effort:** S (~5 min wall clock, mostly waiting on inference)

### TODO-RESUME-004: Pin mlx-vlm in backend/requirements.txt (task #19)
- **What:** Add `mlx-vlm>=0.4.4` to `backend/requirements.txt`. Also document the `pip install --break-system-packages` install pattern (the project uses global Homebrew Python; mlx is hand-installed there too).
- **Effort:** S (~5 min)

### TODO-RESUME-005: Unit tests for delivery_window detector (task #15)
- **What:** Write pytest tests in `backend/tests/test_delivery_window.py`. Cover: clip with clean delivery (Ball outcome), clip with broadcast cut (in-play outcome), clip with no pitcher detected, clip with 60fps source, edge case where set_frame is at frame 0. Use the existing two Darvish CH clips as fixtures (commit them to the test fixtures directory).
- **Effort:** M (~1 hour)

### TODO-FETCH-BUG (rolled forward from morning): fetch_savant_clips.py default angle
- **What:** `scripts/fetch_savant_clips.py` defaults to `--angle AWAY` which returns no MP4 for 2017+ postseason. Make the default fall back through HOME → CMS_NATIONAL → NETWORK → AWAY automatically.
- **Effort:** S (~1 hour, ~10 lines in `fetch_game_clips`)

---

## P1 — Must Do for the Tipping-Confirmation Demo (current focus)

### TODO-DEMO-001: Visual mesh inspection of Darvish CH inference
- **What:** Render `data/meshes/526517/b69a4fbd.npz` to a single Three.js viewer or Blender frame and visually confirm the geometry. Check for left/right joint swap (most common monocular failure mode), confirm the body orientation matches the broadcast frame, confirm joint smoothness across consecutive frames.
- **Why:** Numerical sanity check passed (height matches Darvish, body-center travel matches a delivery), but visual confirmation closes the last verification gap before we commit to overnight batch inference.
- **Effort:** S (15-60 min)
- **Depends on:** Existing Blender or R3F render scripts.

### TODO-DEMO-002: Run full WS G7 batch inference (47 Darvish pitches)
- **What:** `python scripts/fetch_savant_clips.py --pitcher darvish --game-pk 526517 --angle HOME` then `python scripts/batch_inference.py --game-pk 526517`. Approximately 120 minutes wall clock at ~2.5 min/pitch on M3 Max MLX.
- **Why:** All 47 pitches are needed to compare FF (n=13) vs ST (n=17) groups for the tipping demo.
- **Effort:** S to set up, runs unattended overnight
- **Depends on:** TODO-DEMO-001 passes

### TODO-DEMO-003: Run full Sep 19 control batch (97 Darvish pitches)
- **What:** Same workflow on `--game-pk 492355`. Approximately 240 minutes wall clock.
- **Why:** Control game (no allegations) for the binary tipping comparison.
- **Effort:** S to set up, runs unattended
- **Depends on:** TODO-DEMO-002

### TODO-DEMO-004: Finish blender-render spec
- **What:** Implement the side-by-side comparison render with phase-aligned ghost overlay, joint highlight at the maximum-delta frame, and one-sentence on-screen subtitle. The spec is currently `draft` in the manifest.
- **Why:** This is what the demo video will show.
- **Effort:** M (1-2 days)

### TODO-DEMO-005: Record 60-90 second demo video
- **What:** Screen-capture the analyze dashboard typing the Darvish query, watching the pipeline run, the 3D viewer popping up, and the diagnostic narrative. Edit with subtitles and music.
- **Effort:** M (1 day)

### TODO-DEMO-006: Write the companion blog post
- **What:** ~1500 words: "What the numbers say about Darvish in WS Game 7 (and what they can't say)." Honest about validation limits in the closing section. Cross-link to VALIDATION.md.
- **Effort:** M (1 day)

### TODO-DEMO-007: Release pass — LinkedIn, Twitter, r/baseball, MLB cold emails
- **What:** Post the demo video + blog post. Tag Driveline, Pitching Ninja, Eno Sarris, Jomboy. Apply to 3-5 targeted MLB player-dev roles with the demo link in paragraph 1.
- **Effort:** S (4 hours)
- **Depends on:** TODO-DEMO-005, TODO-DEMO-006

### TODO-FETCH-BUG: fetch_savant_clips.py default angle bug
- **What:** `scripts/fetch_savant_clips.py` defaults to `--angle AWAY` which returns no MP4 for 2017 (and likely other pre-2019) postseason play_ids. Make the default automatically fall back through HOME → CMS_NATIONAL → NETWORK → AWAY until one returns a valid MP4 URL.
- **Why:** Discovered during the Darvish 2017 WS G7 verification. Without `--angle HOME` the script silently fails for older games. The bug also affects future researchers using older Statcast data.
- **Effort:** S (~1 hour, ~10-line change in `fetch_game_clips`)

## P2 — Build After the Demo Lands

### TODO-002: Build Statcast pitch-matching engine
- **What:** Cross-reference matching (velocity + pitch type + timing) with sequence alignment fallback (DTW / Needleman-Wunsch) to match video-detected pitches to Statcast records.
- **Why:** This is the novel engineering piece. No open-source tool does video-pitch ↔ Statcast matching. Getting this right determines whether mechanics-to-outcomes correlations are trustworthy.
- **Effort:** M
- **Depends on:** None (can build against assumed SAM4Dcap output schema; validate end-to-end when GPU spike completes)

### TODO-003: Rewrite specs for reframed vision
- **What:** Update manifest, revise ~10 existing specs, add 3 new specs (stream ingestion, broadcast scene detection, companion UI). Align with the Ohtani-first, live-companion + upload dual-mode vision.
- **Why:** Current 18 specs reflect the original upload-only concept. The CEO review reshaped the product.
- **Effort:** M
- **Depends on:** None (proceed with assumed SAM4Dcap output; spike validates quality later)

## P2 — Build During Implementation

### TODO-004: Broadcast pitch segmentation with replay detection
- **What:** Auto-detect individual pitch deliveries from continuous broadcast footage. Skip replays, camera cuts, commercials, and PIP overlays.
- **Why:** Replay false positives produce garbage analysis. Camera cuts mid-delivery lose data. This is the hardest computer vision sub-problem in the pipeline.
- **Effort:** L
- **Depends on:** None (build against assumed output schema; validate with GPU spike later)

### TODO-005: Pitcher transition detection via Statcast game log
- **What:** Use Statcast data to know when the starter (Ohtani) leaves the game. Stop analysis, don't process reliever pitches as if they're Ohtani.
- **Why:** Avoids contaminating Ohtani's analysis with another pitcher's mechanics.
- **Effort:** S

### TODO-006: Confidence scores on all analysis outputs
- **What:** Every analysis result (tipping, fatigue, command, etc.) includes a confidence metric. Reconstruction quality score per frame. Matching confidence per pitch.
- **Why:** Honest uncertainty is the difference between a tool baseball people trust and one they dismiss. "Tipping detected (73% confidence)" is credible.
- **Effort:** M

### TODO-007: NaN/inf handling in feature extraction
- **What:** Guard against NaN/inf in angular velocity/acceleration computations (Savitzky-Golay on noisy data). Propagate quality flags downstream.
- **Why:** One NaN in the kinetic chain analysis crashes the entire pipeline. Silent failure mode.
- **Effort:** S

## Vision Items — Future Phases

### VISION-007: Spike SAM4Dcap on a single Ohtani pitch delivery (was TODO-001)
- **What:** Run SAM4Dcap on one short broadcast clip of an Ohtani pitch to validate the reconstruction pipeline.
- **Why:** The entire project rests on SAM 3D Body producing usable 3D joints from 1080p broadcast footage. If the quality is insufficient, the approach needs rethinking before writing any code.
- **Validate:** (a) produces usable 3D joints, (b) biomechanics output maps to pitching-relevant features, (c) quality from broadcast footage is sufficient.
- **Effort:** S (2-4 hours)
- **Depends on:** NVIDIA GPU access (Colab, Lambda, or similar). SAM4Dcap/SAM-Body4D pipeline requires CUDA — no MPS/Apple Silicon path exists for the full mesh+biomechanics stack.
- **Deferred:** 2026-04-04. No local NVIDIA GPU available (M3 Max only). TODO-002/003/004 decoupled to proceed with assumed output schema.

### VISION-001: Animated kinetic chain energy flow visualization
- **What:** Glowing trail flowing from ground through pelvis → trunk → shoulder → elbow → wrist on the 3D mesh, timed to actual kinetic chain sequence. Sankey diagram on a human body.
- **Why:** This is the screenshot that goes viral on Baseball Twitter. Demo showstopper.
- **Effort:** M (R3F shader work)

### VISION-002: "What changed?" auto-diff
- **What:** Compare two games from different dates. One-button mechanical comparison: arm slot delta, hip-shoulder sep delta, stride length delta.
- **Why:** Shows the tool's value over time. Low effort since analysis engine already computes these.
- **Effort:** S

### VISION-003: Shareable comparison GIFs
- **What:** Export two synced deliveries (fastball vs sweeper) as animated GIF. Headless browser capture + ffmpeg.
- **Why:** Baseball Twitter lives on this format. Free marketing.
- **Effort:** S

### VISION-004: MLX Apple Silicon port of SAM 3D Body
- **What:** Port SAM 3D Body to MLX as `sam3d_mlx/` module within SamPlaysBaseball.
- **Plan:** `docs/plans/specs/mlx-port-manifest.md` (6 specs, 3 phases)
- **Status:** Planned — specs reviewed via /plan-eng-review (2026-04-03)
- **Why:** Local inference, no cloud costs, faster iteration. Impressive open-source contribution.
- **Effort:** XL
- **Key findings from eng review:** backbone spec had wrong architecture params (now fixed), roma dependency unaccounted for (now in mhr_utils), TurboQuant made optional

### TODO-008: Generate parity test fixtures from PyTorch model
- **What:** Run PyTorch SAM 3D Body on 3-5 test inputs, save intermediate outputs (backbone features, decoder tokens, MHR vertices, mhr_utils conversion results) as .npz files.
- **Why:** MLX parity tests need ground truth. Without pre-generated fixtures, tests require PyTorch installed alongside MLX, defeating the zero-PyTorch goal.
- **Effort:** S (2-4 hours)
- **Depends on:** GPU access (or CPU with patience), PyTorch environment

### VISION-005: Multi-pitcher comparison database
- **What:** Expand beyond Ohtani. Process 5-10 pitchers. Population-level comparison.
- **Why:** "Your arm slot is in the 85th percentile" is more useful than "your arm slot is X degrees."
- **Effort:** M (pipeline is built, just needs more data)

### VISION-006: Live stream companion mode
- **What:** Real-time analysis while watching a game. Stream ingestion, per-pitch alerts, companion UI.
- **Why:** The ultimate product vision. Requires Fast SAM 3D Body + persistent GPU.
- **Effort:** XL
