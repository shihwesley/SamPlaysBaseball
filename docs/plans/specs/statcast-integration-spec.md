---
name: statcast-integration
phase: 1
sprint: 1
parent: data-model
depends_on: [data-model]
status: implemented
created: 2026-02-16
updated: 2026-04-04
---

# Statcast Integration Spec

Connects 3D biomechanics data to Baseball Savant pitch outcomes — closing the loop from "how did the pitcher move" to "what happened to the pitch."

## Implementation

### StatcastFetcher (backend/app/data/statcast.py)

```python
fetcher = StatcastFetcher(cache_dir="data/cache")
df = fetcher.fetch_pitcher(660271, "2025-10-01", "2025-11-01")  # pybaseball
df = fetcher.load_csv("savant_export.csv")                       # CSV import
row = fetcher.match_pitch(df, pitch_metadata)                    # exact key match
results = fetcher.match_all(df, list_of_metadata)                # batch match
```

Fetch modes: pybaseball (live API) or CSV import. Caches as Parquet. Retains columns: release_speed, release_spin_rate, pfx_x/z, plate_x/z, spin_axis, vx0/vy0/vz0, ax/ay/az, woba_value, estimated_woba, launch_speed/angle, effective_speed, release_extension.

Simple pitch matching: `game_date + pitcher_id + inning + pitch_number`.

### PitchDB Enrichment (backend/app/data/pitch_db.py)

```python
db = PitchDB("data/pitches.db")
db.enrich_from_statcast(game_pk=813024, pitcher_id=660271)
```

Fetches Statcast CSV for a game date, matches by inning/AB/pitch order, merges velocity, spin, movement, location, and outcome metrics into existing SQLite rows. Resolves game_date from MLB Stats API if not in DB.

### Baseball Savant play_id Linkage

For Savant-sourced clips, pitch matching is solved by design — each clip's `play_id` maps directly to its Statcast row via the `gf?game_pk=` endpoint. The DTW/sequence alignment matcher (TODO-002 design) becomes a fallback for non-Savant video sources only.

### Correlation Engine (backend/app/analysis/correlation.py)

```python
engine = CorrelationEngine()
corr = engine.correlate_all(features_df, outcomes_df)      # Pearson + Spearman
reg = engine.regress_all(features_df, outcomes_df)          # Ridge/LASSO
scatter = engine.scatter_data(features_df, outcomes_df, "arm_slot", "release_speed")
```

- Correlation: per-feature × per-outcome, both overall and per pitch type (FB/SL/CB/CH)
- Regression: Ridge or LASSO with ranked features by |coefficient|
- Scatter data: x/y points + regression line for visualization
- Outcome metrics: release_speed, release_spin_rate, woba_value, estimated_woba, launch_speed/angle, effective_speed, release_extension

### Player Search (backend/app/data/player_search.py)

```python
ps = PlayerSearch()
players = ps.search_pitcher("yamamoto")                    # name → MLBAM ID
games = ps.pitching_games(808967, season=2025)             # season game log
pitches = ps.game_pitches(game_pk, pitcher_id)             # per-pitch play_ids
fastballs = ps.game_pitches_by_type(game_pk, pid, "FF")   # filtered
```

Uses MLB Stats API for player/game discovery, Baseball Savant `gf` endpoint for per-pitch data.

## Validated

- Ohtani WS Game 7: 50/52 pitches enriched with full Statcast metrics
- Player search: "yamamoto" → Yoshinobu Yamamoto (ID: 808967), 30 games in 2025
- Correlation engine: tested with known relationships, NaN handling, per-pitch-type grouping

## Planned: Tiered Pitch Matcher (TODO-002)

Design complete at `docs/plans/2026-04-04-pitch-matcher-design.md`. Three tiers:
1. Exact key match (inning + pitch_number) — current implementation
2. DTW sequence alignment by velocity — for broadcast footage with missed pitches
3. Nearest neighbor fallback — low confidence

Blocked on feature extraction producing `ObservedPitch` data from video pipeline. Not needed for Savant-sourced clips.

## Files

| File | Purpose | Status |
|------|---------|--------|
| backend/app/data/statcast.py | StatcastFetcher (pybaseball + CSV + matching) | implemented |
| backend/app/data/pitch_db.py | PitchDB.enrich_from_statcast() | implemented |
| backend/app/data/player_search.py | MLB Stats API player/game search | implemented |
| backend/app/analysis/correlation.py | CorrelationEngine (Pearson/Spearman/Ridge) | implemented |
| backend/tests/test_statcast.py | StatcastFetcher + CorrelationEngine tests | implemented |
| backend/app/data/pitch_matcher.py | Tiered DTW matcher | designed (TODO-002) |

## Dependencies

- Upstream: data-model (PitchMetadata, PitchDB)
- Downstream: api-layer, ai-scouting-reports, pitch-matcher
