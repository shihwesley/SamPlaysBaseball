# TODOS

## P1 — Must Do Before Building

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
