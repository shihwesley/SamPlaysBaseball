# Mechanics Diagnostic Tool — Design Doc

**Date:** 2026-04-05
**Status:** Design approved, ready for implementation
**Scope:** Phase 3 integration — query-driven pitcher mechanics analysis

## Problem

A player development analyst notices something wrong with a pitcher during or after a game. Maybe command fell off, velocity dropped, or a pitch is getting hit harder than usual. Today they watch video manually, eyeball mechanics, and rely on intuition. They want biomechanical data backing up what they see.

## Product

The analyst types a natural language query. The app pulls the relevant footage and Statcast data, runs 3D body reconstruction via MLX, extracts biomechanical features, compares deliveries, and produces an expert diagnostic report with an interactive 3D viewer.

One query, one response bundle. The dashboard fills in with everything the analyst needs to brief the pitching coach.

## Users

MLB player development staff: pitching coaches, biomechanics analysts, front office evaluators. They know pitching mechanics terminology. They don't need hand-holding on what shoulder ER means — they need the numbers faster and with 3D evidence.

## Architecture

```
Query bar (natural language)
    │
    ▼
/api/query/parse — LLM structured extraction
    │
    ▼
/api/query — Orchestration endpoint
    ├── Resolve pitcher (PlayerSearch)
    ├── Fetch pitches (PitchDB, filtered)
    ├── Check mesh cache / queue MLX inference
    ├── Run compare_deliveries()
    ├── Gather Statcast context
    ├── Run LLM diagnostic report
    └── Export comparison GLB
    │
    ▼
JSON response bundle → Dashboard renders all panels
```

Single orchestration endpoint returns everything. Frontend makes one call. For uncached pitches, returns a progress token; frontend polls until inference completes.

## Query Parser

Takes free text, returns structured params. Single LLM call with the available pitchers and pitch types from the DB as grounding context.

### Input

```
"Compare Ohtani's first inning fastballs to his 6th inning fastballs"
"Why did his slider command fall off after the 4th?"
"Show me his cutter vs fastball delivery"
```

### Output

```python
@dataclass
class AnalysisQuery:
    pitcher_name: str               # "Ohtani"
    pitch_types: list[str]          # ["FF"] or ["FF", "FC"]
    comparison_mode: str            # "time" | "type" | "baseline"
    inning_range_a: tuple | None    # (1, 3)
    inning_range_b: tuple | None    # (5, 7)
    game_date: str | None           # "2026-04-01" or None (most recent)
    concern: str | None             # "command" | "velocity" | "getting hit"
```

### Comparison modes

- **time** — Same pitch type, different innings (early vs late). Detects fatigue, mechanical drift.
- **type** — Different pitch types, same game (FF vs FC). Detects tipping, tunneling issues.
- **baseline** — Current pitch vs pitcher's computed baseline (20+ pitch average). Detects chronic changes.

### Parser prompt grounding

The prompt receives the list of pitchers in the DB and their pitch type inventory so it doesn't hallucinate names or types. Example: `Available pitchers: Ohtani (FF, SL, CU, FC, SP). Available games: 2026-04-01 (game_pk 813024).`

## Orchestration Flow

`POST /api/query`

```
1. Parse query → AnalysisQuery
2. Resolve pitcher_name → pitcher_id (PlayerSearch)
3. Fetch pitches from PitchDB
   - Filter: pitch_type, game_pk (from game_date), inning range
   - Split into group A and group B per comparison_mode
4. For each pitch: check mesh_path exists in DB
   - Cached → load mesh .npz
   - Not cached → run MLX inference, store result, return progress token
5. Select representative pitch from each group
   - time mode: median pitch by inning from each range
   - type mode: one sample of each type
   - baseline mode: latest pitch vs computed baseline
6. compare_deliveries(joints_a, joints_b, ...)
7. Gather Statcast aggregates per group (velo, spin, whiff%, zone%)
8. Build LLM context dict, call diagnostic report generator
9. Export comparison GLB via ComparisonGLBBuilder
10. Return JSON bundle
```

### Response bundle

```json
{
  "report": {
    "narrative": "4-6 paragraph expert assessment...",
    "recommendations": ["Monitor trunk rotation...", ...],
    "risk_flags": ["Reduced hip-shoulder separation below 25 deg threshold"],
    "confidence": "high",
    "pitches_analyzed": 21
  },
  "comparison": {
    "label_a": "FF Inn 1-3",
    "label_b": "FF Inn 5-7",
    "diffs": [...],
    "phase_timing": {...},
    "kinetic_chain_a": "pelvis → trunk → shoulder → elbow → wrist",
    "kinetic_chain_b": "pelvis → trunk → shoulder → wrist → elbow"
  },
  "statcast": {
    "group_a": {"avg_velo": 97.2, "avg_spin": 2412, "whiff_pct": 32, "zone_pct": 48},
    "group_b": {"avg_velo": 95.1, "avg_spin": 2389, "whiff_pct": 19, "zone_pct": 31}
  },
  "viewer": {
    "glb_url": "/api/export/glb/{comparison_id}",
    "phase_markers": {"foot_plant": 37, "mer": 78, "release": 119},
    "total_frames": 120
  },
  "query": {
    "parsed": {...},
    "pitches_used": ["play_id_1", "play_id_2"]
  }
}
```

### Progress token (uncached pitches)

When MLX inference is needed:

```json
{
  "status": "processing",
  "token": "q_abc123",
  "progress": {"current_frame": 47, "total_frames": 370, "pitch_index": 1, "total_pitches": 2},
  "eta_seconds": 180
}
```

Frontend polls `GET /api/query/{token}/status` until status flips to `"complete"`, then fetches the full bundle.

## LLM Diagnostic Report

### Model: Gemma 4 E4B (local, multimodal)

Runs locally on Apple Silicon via mlx-vlm. 4B parameters, ~16GB memory. The key advantage: SigLIP2 vision encoder means the model can *see* rendered frames of the delivery, not just read numbers. Zero API cost, zero latency to external services, all data stays on the machine.

```
Model: google/gemma-4-e4b-it
Runtime: mlx-vlm (Apple Silicon native)
Memory: ~16GB (alongside SAM 3D Body model on M3 Max 128GB)
Capabilities: vision (SigLIP2) + text + thinking mode
```

Fallback: Claude API via existing LLMReportGenerator for higher-quality prose when needed. The interface is the same — both produce DiagnosticReport.

### Visual input (new — enabled by multimodal)

Before calling the LLM, render 3-6 images from the 3D scene:

1. **Key frame triptych** — Side-by-side renders of foot plant, MER, and release for pitch A. Same for pitch B. Two rows, three columns. Annotated with frame numbers and phase labels.
2. **Ghost overlay at MER** — Both pitches overlaid (solid + transparent) at the moment of maximum external rotation. Catcher's view. This is where arm slot differences are most visible.
3. **Ghost overlay at release** — Same overlay at ball release. Shows release point and trunk position differences.
4. **Skeleton trajectory** — Top-down view of the wrist path through space for both pitches. Colored trails (blue = pitch A, orange = pitch B). Shows arm path divergence.

These renders come from the existing Three.js components (MoundScene, PitcherMesh, GhostOverlay, SkeletonOverlay) via server-side rendering or headless screenshot. For MVP, use matplotlib or trimesh to render static frames from the joint arrays — no browser needed.

### Role

The LLM acts as a 30-year veteran pitching mechanics analyst. Not a chatbot — a colleague writing an internal assessment. No hedging, no filler, specific numbers, scout terminology.

### System prompt

```
You are a pitching mechanics analyst with 30 years of professional
experience across MLB player development. You have worked with dozens
of pitchers from rookie ball to the majors. You are writing an internal
assessment for the pitching coach and player development staff.

You are looking at rendered 3D body reconstructions of the pitcher's
delivery. The images show the body mesh and skeleton at key phases
(foot plant, MER, release). When a ghost overlay is shown, the solid
body is the primary pitch and the transparent body is the comparison.

Rules:
- Describe what you SEE in the images first, then connect to the numbers
- Lead with what changed mechanically and why it matters for performance
- Connect mechanical changes to the Statcast outcomes the analyst sees
- Reference specific numbers — never generalize
- Use biomechanics terminology (MER, hip-shoulder separation, kinetic
  chain sequencing) but explain the implication in plain terms
- If the kinetic chain is disrupted, say so explicitly
- Flag injury risk when relevant (inverted W, low shoulder ER with
  high velocity = UCL stress, reduced hip-shoulder sep = arm-dominant)
- End with 3-5 specific, actionable items for the pitching coach
- Write 4-6 paragraphs. No headers, no bullets in the narrative.
  Recommendations as a numbered list at the end.
- Never say "based on the data" or "the analysis shows" — state
  findings directly as observations
```

### Context (multimodal message: images + text)

The prompt combines rendered images with a structured text block:

**Images** (3-6, passed as image tokens):
- Key frame triptych (pitch A and pitch B)
- Ghost overlay at MER
- Ghost overlay at release
- Wrist trajectory comparison (optional)

**Text block:**

```python
{
    "concern": str,                 # analyst's original question
    "pitcher": str,
    "handedness": str,
    "comparison": str,              # what's being compared (readable)
    "label_a": str,
    "label_b": str,
    "mechanical_changes": [         # from DeliveryComparison.diffs, sorted by magnitude
        {"name": str, "early": float, "late": float, "change": float, "unit": str},
    ],
    "norms": {                      # normative ranges from biomechanics literature
        "max_shoulder_er_deg": {"healthy": [150, 180], "concern_below": 140},
        "hip_shoulder_sep_deg": {"healthy": [35, 60], "concern_below": 25},
        "stride_length_normalized": {"healthy": [0.75, 0.90]},
        "elbow_flexion_vel": {"healthy": [2000, 2800], "unit": "deg/s"},
        "trunk_rotation_range": {"healthy": [25, 50], "concern_below": 15},
    },
    "kinetic_chain_a": str,
    "kinetic_chain_b": str,
    "phase_timing": dict,
    "peak_divergence": str,         # where in the delivery they differ most
    "statcast": {
        "velo_a": float, "velo_b": float,
        "spin_a": float, "spin_b": float,
        "whiff_pct_a": float, "whiff_pct_b": float,
        "zone_pct_a": float, "zone_pct_b": float,
    },
}
```

### Output model

```python
@dataclass
class DiagnosticReport:
    narrative: str              # 4-6 paragraph expert assessment
    recommendations: list[str]  # 3-5 actionable items
    risk_flags: list[str]       # injury/performance concerns (may be empty)
    confidence: str             # "high" | "moderate" | "low"
    pitches_analyzed: int
```

### Confidence rules

- < 3 pitches in either group → don't generate report, return error
- 3-4 pitches → "low" confidence, report includes caveat
- 5-19 pitches → "moderate"
- 20+ pitches → "high"

### Normative ranges source

Compiled from:
- Driveline OpenBiomechanics dataset (100K+ pitches)
- American Sports Medicine Institute (ASMI) published ranges
- Fleisig et al. (2011) — kinematic parameters of elite pitchers
- Pitcher-specific baseline when available (20+ pitch history)

These are baked into the context dict, not the system prompt. They evolve as we get more data.

## 3D Viewer

### Scene composition

- MLB regulation mound (18ft diameter, 10.5in height, 6ft slope) — `mound.py` generates geometry
- Pitcher mesh placed on rubber, ground-aligned via `GroundPlaneAligner`, height-scaled
- Optional ghost overlay (transparent blue, 30% opacity) for comparison pitch
- Ambient + directional lighting, dark background

### Existing components (frontend/src/components/three/)

| Component | Lines | Status | Notes |
|-----------|-------|--------|-------|
| MoundScene | 46 | Working | Canvas + mound cylinder + orbit controls + camera presets |
| PitcherMesh | 53 | Working | GLB loader with morph target frame control |
| GhostOverlay | 43 | Working | Cloned scene with transparent material |
| TimelineScrubber | 88 | Working | Play/pause, scrub, phase markers |
| CameraPresets | 35 | Working | Catcher, 1B, 3B, overhead, behind-pitcher |
| SkeletonOverlay | 45 | Working | Joint wireframe on top of mesh |
| DeviationColoring | 59 | Working | Per-vertex coloring by deviation magnitude |
| SplitSync | 57 | Working | Side-by-side synced playback |
| Stroboscope | 36 | Working | Multi-frame freeze (foot plant + MER + release) |

### New components needed

| Component | Purpose |
|-----------|---------|
| JointSelector | Click detection on mesh joints, highlights selected joint |
| MetricGraph | Time-series chart for selected joint (angle + velocity, both pitches) |
| SpeedControl | 0.25x, 0.5x, 1x playback speed buttons |
| FieldGeometry | Static baseball field (grass, dirt, baselines) — low-poly, not photorealistic |
| QueryBar | Text input + filter chips, calls /api/query |
| ReportPanel | Renders DiagnosticReport narrative + recommendations + risk flags |
| MetricsPanel | Sorted diff table with sparklines |
| StatcastPanel | Group A vs Group B performance stats |

## Inference Timing Model

### MVP (Ohtani-first, on-demand with cache)

Analyst queries a pitch. If mesh exists in PitchDB, instant. If not, MLX inference runs at ~490ms/frame. A 370-frame clip (6s at 60fps) takes ~3 minutes. Progress streamed to frontend.

Results cached permanently in PitchDB as .npz. Second query for same pitch is instant.

### Production (post-MVP)

After each game, the team uploads footage. A batch job runs `batch_inference.py --backend mlx --game-pk {id}`. All pitches processed overnight. By morning, everything is cached.

Historical footage (prior seasons, other pitchers) stays on-demand with cache.

## Data Flow Summary

```
Game footage (MP4 clips)
    │
    ▼
fetch_savant_clips.py → PitchDB (metadata + video paths)
    │
    ▼
batch_inference.py --backend mlx → PitchDB (mesh .npz cached)
    │
    ▼
Analyst query → /api/query
    │
    ├── PitchDB.load_mesh() → joints_mhr70 (T, 70, 3)
    ├── FeatureExtractor.extract_from_array() → BiomechFeatures
    ├── compare_deliveries() → DeliveryComparison
    ├── Statcast aggregation
    ├── LLM diagnostic → DiagnosticReport
    └── GLB export → comparison .glb file
    │
    ▼
Dashboard (3D viewer + report + metrics + Statcast)
```

## Files to Create/Modify

### New files

| File | Purpose |
|------|---------|
| backend/app/api/query.py | Orchestration endpoint — /api/query, /api/query/{token}/status |
| backend/app/query/parser.py | LLM query parser → AnalysisQuery |
| backend/app/query/orchestrator.py | Chains parser → fetch → compare → report → GLB |
| backend/app/reports/diagnostic.py | Diagnostic report generator (wraps LLMReportGenerator with mechanics-specific prompt and normative ranges) |
| backend/app/reports/norms.py | Normative biomechanical ranges dict |
| backend/tests/test_query_parser.py | Parser tests with example queries |
| backend/tests/test_diagnostic.py | Report generation tests (mocked LLM) |
| frontend/src/app/analyze/page.tsx | Main dashboard page |
| frontend/src/components/QueryBar.tsx | Query input + filter chips |
| frontend/src/components/ReportPanel.tsx | Diagnostic report display |
| frontend/src/components/MetricsPanel.tsx | Feature diff table |
| frontend/src/components/three/JointSelector.tsx | Click-to-select joint on mesh |
| frontend/src/components/three/MetricGraph.tsx | Joint angle/velocity time series |
| frontend/src/components/three/SpeedControl.tsx | Playback speed buttons |
| frontend/src/components/three/FieldGeometry.tsx | Static baseball field geometry |

| backend/app/reports/renderer.py | Render key frame images from joint arrays (matplotlib/trimesh, no browser needed) |
| backend/app/reports/vlm.py | Gemma 4 E4B integration via mlx-vlm — multimodal diagnostic generation |

### Modified files

| File | Change |
|------|--------|
| backend/app/reports/llm.py | Add generate_diagnostic() method, keep as Claude API fallback |
| backend/app/api/routes.py | Register /api/query routes |
| frontend/src/app/layout.tsx | Add /analyze route to nav |

## Implementation Order

1. **Normative ranges + diagnostic report generator** — backend/app/reports/diagnostic.py, norms.py. Can test with mocked comparison data immediately.
2. **Query parser** — backend/app/query/parser.py. LLM structured extraction. Test with example queries.
3. **Orchestration endpoint** — backend/app/query/orchestrator.py + api/query.py. Wires everything together.
4. **Dashboard page** — frontend/src/app/analyze/page.tsx. Wire existing Three.js components + new panels.
5. **Joint click + metric graph** — JointSelector + MetricGraph. Enhancement after core flow works.
6. **Field geometry** — FieldGeometry.tsx. Visual polish, not blocking.

Steps 1-3 are backend-only and can be built and tested without touching the frontend. Step 4 is assembly — the individual components already exist.
