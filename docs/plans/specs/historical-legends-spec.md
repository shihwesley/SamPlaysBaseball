---
name: historical-legends
phase: 4
sprint: 1
parent: null
depends_on: [video-pipeline, sam3d-inference, feature-extraction, baseline-comparison]
status: draft
created: 2026-02-16
---

# Historical Legends Analysis Spec

Process old broadcast footage of iconic pitchers and compare their mechanics. Only a single-camera tool like SAM 3D Body can do this — KinaTrax didn't exist in 1995.

## Requirements

- Process broadcast footage of 3-5 legendary pitchers
- Run full analysis pipeline on their mechanics
- Enable comparison: legend vs current pitcher
- Curated talking points for each legend

## Acceptance Criteria

- [ ] 3-5 legendary pitchers processed from available broadcast footage:
  - Mariano Rivera (cutter mechanics — the most unhittable pitch ever)
  - Pedro Martinez (changeup deception — different arm speed, same slot)
  - Clayton Kershaw (curveball mechanics — elite curve at consistent slot)
  - Greg Maddux (command king — release point consistency)
  - Randy Johnson (extreme arm slot + stride length)
- [ ] Full pipeline: video → SAM3D → features → analysis for each
- [ ] "Compare to legend" feature: overlay current pitcher's mechanics against a legend
- [ ] Talking points per legend: what made their mechanics special, in biomechanical terms
- [ ] Honest accuracy disclaimer: broadcast footage from 1990s/2000s is lower quality, SAM3D accuracy will be lower
- [ ] Pre-computed and included in demo data

## Technical Approach

Source footage from YouTube (fair use for research/education — non-commercial project). Process through the same pipeline as any other video. Key challenge: lower resolution footage will reduce SAM 3D Body accuracy. Mitigate by using super-resolution preprocessing (Real-ESRGAN or similar) before SAM3D inference.

The comparison feature is powerful in demos: "Here's your pitcher's changeup delivery overlaid with Pedro Martinez's. Notice the arm speed difference at frame 22 — Pedro's deception comes from maintaining identical arm speed through cocking phase."

This is a conversation starter. Every baseball person has opinions about these pitchers. When your tool can show *why* they were great in biomechanical terms, it demonstrates both technical capability and baseball knowledge.

## Files

| File | Purpose |
|------|---------|
| demo/legends/process_legends.py | Legend video processing script |
| demo/legends/talking_points.json | Per-legend analysis notes |
| demo/data/legends/ | Pre-computed legend data |

## Tasks

1. Source and acquire broadcast footage for each legend (YouTube / archive)
2. Preprocess lower-quality footage (super-resolution if needed)
3. Process through full pipeline
4. Generate comparison data (legend vs modern pitcher overlays)
5. Write talking points per legend (biomechanical analysis of what made them special)

## Dependencies

- Upstream: video-pipeline, sam3d-inference, feature-extraction, baseline-comparison
- Downstream: demo-mode
