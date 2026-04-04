---
name: data-model
phase: 1
sprint: 1
parent: root
depends_on: []
status: implemented
created: 2026-02-16
updated: 2026-04-04
---

# Data Model Spec

The foundation layer. Every other spec depends on these structures.

## Requirements

- Core data structures for pitch data, 3D mesh/skeleton output, and analysis results
- Storage layer for both live processing and pre-computed demo mode
- Serialization for passing data between pipeline stages and to the API layer
- Player search and game discovery from MLB APIs

## Implementation

### Pitch Models (backend/app/models/pitch.py)

```python
PitchMetadata    # pitcher_id, game_date, inning, pitch_number, pitch_type, velocity, spin, location, video_path
PitchData        # metadata + joints (T,127,3), joints_mhr70 (T,70,3), pose_params (T,136), shape_params (45,)
```

Both are Pydantic models with numpy conversion helpers (`joints_array()`, `from_numpy()`).

### Database Models (backend/app/data/pitch_db.py)

```python
PitchRecord      # flat dataclass for SQLite row — play_id, game_pk, pitcher_id, pitch_type,
                 # release_speed, spin_rate, movement (pfx_x/z), location (plate_x/z),
                 # outcomes (woba, launch_speed/angle), video_path, mesh_path

MeshData         # numpy arrays for one pitch clip:
                 #   vertices (T, 18439, 3), joints_mhr70 (T, 70, 3),
                 #   pose_params (T, 136), shape_params (45,),
                 #   cam_t (T, 3), focal_length (float)

PitchDB          # SQLite + .npz storage manager
                 #   insert_pitch(), insert_from_manifest(), update_mesh(),
                 #   enrich_from_statcast(), query(), load_mesh(), summary()
```

### Fetcher Models (scripts/fetch_savant_clips.py)

```python
PitchClip        # play_id, game_pk, pitcher/batter names, inning, pitch_type,
                 # video_url, video_path — one per downloaded clip
```

### Player Search (backend/app/data/player_search.py)

```python
Player           # id, name, position, team
GameAppearance   # game_pk, date, opponent, IP, K, ER, BB, H
PitchInfo        # play_id, inning, at_bat_number, pitch_type, speed, description
```

### Storage Architecture

- **SQLite** (`data/pitches.db`) — pitch metadata + Statcast metrics, queryable
- **.npz files** (`data/meshes/{game_pk}/{play_id[:8]}.npz`) — heavy 3D arrays per pitch
- **Manifest JSON** (`data/clips/{game_pk}/manifest.json`) — fetch results per game

### Existing Analysis Models (backend/app/analysis/correlation.py)

`CorrelationEngine` produces correlation results and regression results. These feed into the scouting reports module. Model definitions for the 6 analysis modules (baseline, tipping, fatigue, command, arm-slot, timing) are deferred to the feature-extraction spec.

## Files

| File | Purpose | Status |
|------|---------|--------|
| backend/app/models/pitch.py | PitchMetadata, PitchData | implemented |
| backend/app/data/pitch_db.py | PitchDB, PitchRecord, MeshData | implemented |
| backend/app/data/player_search.py | Player, GameAppearance, PitchInfo | implemented |
| scripts/fetch_savant_clips.py | PitchClip | implemented |
| backend/app/models/baseline.py | PitcherBaseline | planned |
| backend/app/models/analysis.py | AnalysisResult variants per module | planned |
| backend/tests/test_models.py | Model validation tests | exists (partial) |

## Dependencies

- Upstream: none (root spec)
- Downstream: video-pipeline, sam3d-inference, feature-extraction, api-layer, statcast-integration
