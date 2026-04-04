---
name: video-pipeline
phase: 1
sprint: 2
parent: data-model
depends_on: [data-model]
status: draft
created: 2026-02-16
---

# Video Pipeline Spec

> **⚠ NEEDS REVISION (2026-04-04)**
> Architecture changed fundamentally. Spec describes FFmpeg frame extraction from long videos. Actual pipeline uses Baseball Savant per-pitch clips (~6s each, pre-segmented). No pitch segmentation needed. See `scripts/fetch_savant_clips.py` and `backend/app/data/player_search.py`.

Handles video ingestion from all four sources: bullpen camera, broadcast, smartphone, and MiLB footage.

## Requirements

- Accept video files (.mp4, .mov, .avi) from any of the 4 source types
- Extract frames at configurable FPS (30/60)
- Handle source-specific preprocessing (crop broadcast overlays, stabilize phone video)
- Detect and isolate the pitcher when multiple people are in frame
- Output clean, ordered frames ready for SAM 3D Body inference

## Acceptance Criteria

- [ ] Frame extraction works for all supported formats via FFmpeg + OpenCV
- [ ] Bullpen video: direct extraction, minimal preprocessing
- [ ] Broadcast video: crop scoreboard/overlay regions, handle camera cuts
- [ ] Phone video: stabilization pass, handle variable frame rates
- [ ] Pitcher isolation: bounding box filtering when batter/catcher visible
- [ ] Configurable FPS extraction (default 60fps)
- [ ] Pitch segmentation: detect start/end of individual pitch deliveries
- [ ] Output format: ordered frames + metadata (source type, fps, frame count)

## Technical Approach

FFmpeg for frame extraction (subprocess calls — more reliable than pure OpenCV for format handling). OpenCV for preprocessing (crop, stabilize). For broadcast footage, use frame differencing to detect camera cuts and skip non-pitcher frames. Pitcher isolation via SAM 3's built-in detector — take the detection box closest to the mound region.

Pitch segmentation: detect motion onset (background subtraction or optical flow threshold) for delivery start, detect follow-through completion for delivery end. This isolates individual pitches from continuous footage.

## Files

| File | Purpose |
|------|---------|
| backend/app/pipeline/video.py | FrameExtractor class, multi-source handler |
| backend/app/pipeline/preprocess.py | Source-specific preprocessing (crop, stabilize) |
| backend/app/pipeline/segment.py | Pitch delivery segmentation |
| backend/app/pipeline/isolate.py | Pitcher bounding box isolation |
| backend/tests/test_video_pipeline.py | Pipeline tests with sample clips |

## Tasks

1. Implement FFmpeg-based frame extraction with configurable FPS
2. Build multi-source preprocessor (bullpen, broadcast, phone, MiLB handlers)
3. Implement pitcher isolation from multi-person frames
4. Build pitch delivery segmentation (detect start/end of each pitch)
5. Integration test with sample videos from each source type

## Dependencies

- Upstream: data-model (output metadata format)
- Downstream: sam3d-inference consumes extracted frames
