---
name: video-pipeline
phase: 1
sprint: 2
parent: data-model
depends_on: [data-model, statcast-integration]
status: implemented
created: 2026-02-16
updated: 2026-04-04
---

# Video Pipeline Spec

Handles video ingestion — finding pitchers, discovering games, and downloading per-pitch video clips.

## Architecture Change (2026-04-04)

Original spec described FFmpeg frame extraction from long videos with pitch segmentation. That approach was replaced when we discovered Baseball Savant provides **per-pitch video clips** (~6s each, 720p, 60fps) linked directly to Statcast data via `play_id`. This eliminates pitch segmentation, camera cut detection, and replay filtering for Savant-sourced footage.

The old approach (FFmpeg extraction, pitch segmentation, replay detection) remains relevant for custom video sources (bullpen footage, phone recordings) but is deferred to TODO-004.

## Requirements

- Search MLB pitchers by name, discover their game logs
- Download per-pitch video clips from Baseball Savant
- Filter by pitch type, game, season, date range
- Store clips with metadata manifest linking each clip to its Statcast pitch
- Handle Cloudflare-protected CDN via yt-dlp

## Implementation

### Player Search (backend/app/data/player_search.py)

```python
PlayerSearch.search("yamamoto")           # → [Player(id=808967, name="Yoshinobu Yamamoto")]
PlayerSearch.search_pitcher("yamamoto")    # → pitchers only
PlayerSearch.pitching_games(808967, 2025)  # → [GameAppearance(game_pk, date, opponent, IP, K, ...)]
PlayerSearch.game_pitches(game_pk, pid)    # → [PitchInfo(play_id, pitch_type, speed, batter, ...)]
PlayerSearch.game_pitches_by_type(gp, pid, "FF")  # → fastballs only
```

Uses MLB Stats API (no auth, no rate limit) for player search and game logs. Uses Baseball Savant `gf?game_pk=` endpoint for per-pitch `play_id` UUIDs.

### Clip Fetcher (scripts/fetch_savant_clips.py)

```bash
# By name + season (interactive game picker)
python scripts/fetch_savant_clips.py --pitcher yamamoto --season 2025

# By name + specific game
python scripts/fetch_savant_clips.py --pitcher ohtani --game-pk 813024

# Filter to fastballs only
python scripts/fetch_savant_clips.py --pitcher ohtani --game-pk 813024 --pitch-type FF

# By date range
python scripts/fetch_savant_clips.py --pitcher ohtani --start 2025-10-28 --end 2025-11-01
```

Downloads via yt-dlp (Cloudflare blocks direct curl). Validates downloaded files are video (not HTML error pages). Saves manifest.json per game with play_id → clip path mapping.

### Output

```
data/clips/{game_pk}/
├── manifest.json                              # play_id + pitch metadata per clip
├── inn1_ab5_p1_CU_02ec65f0.mp4               # individual pitch clip
├── inn1_ab5_p2_FF_26245b33.mp4
└── ...
```

Each clip: ~6s, 1280x720, 60fps, ~6MB. Naming: `inn{N}_ab{N}_p{N}_{type}_{play_id[:8]}.mp4`

## Validated

- Ohtani WS Game 7 (game_pk=813024): 52 pitches, 51 downloaded (1 UN type has no video)
- yt-dlp handles Cloudflare-protected sporty-clips.mlb.com CDN
- Clips are pre-segmented — no pitch detection needed

## Future: Custom Video Sources (TODO-004)

For non-Savant footage (bullpen, phone, broadcast recordings without Statcast links):
- FFmpeg frame extraction at configurable FPS
- Optical flow / background subtraction for pitch delivery segmentation
- Camera cut and replay detection for broadcast footage
- Pitcher isolation via bounding box filtering

This is a separate implementation track, not blocking the Ohtani MVP.

## Files

| File | Purpose | Status |
|------|---------|--------|
| scripts/fetch_savant_clips.py | Per-pitch clip downloader with name search | implemented |
| backend/app/data/player_search.py | MLB Stats API player/game discovery | implemented |
| backend/app/pipeline/video.py | Custom video frame extraction | planned (TODO-004) |
| backend/app/pipeline/segment.py | Pitch delivery segmentation | planned (TODO-004) |

## Dependencies

- Upstream: data-model (PitchClip, PitchRecord), statcast-integration (game feed API)
- Downstream: sam3d-inference consumes downloaded clips
