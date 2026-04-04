---
name: statcast-integration
phase: 2
sprint: 1
parent: data-model
depends_on: [data-model]
status: draft
created: 2026-02-16
---

# Statcast Integration Spec

> **⚠ NEEDS REVISION (2026-04-04)**
> `StatcastFetcher` implemented and tested per spec. New additions not in spec: `PitchDB.enrich_from_statcast()` merges Statcast CSV data into SQLite. `PlayerSearch` class handles MLB Stats API player/game lookup. Baseball Savant `play_id` linkage eliminates need for complex pitch matching on Savant-sourced data.

Connects your 3D biomechanics data to Baseball Savant pitch outcomes — closing the loop from "how did the pitcher move" to "what happened to the pitch."

## Requirements

- Fetch pitch-level Statcast data from Baseball Savant
- Match Statcast pitches to SAM 3D Body pitches (by date, pitcher, pitch number)
- Correlation analysis: mechanics → outcomes (velocity, spin, movement, location)
- "Mechanics that matter" ranking: which biomechanical features explain the most outcome variance

## Acceptance Criteria

- [ ] Statcast fetcher: pybaseball or direct CSV download from Baseball Savant
- [ ] Data: velocity, spin_rate, spin_axis, pfx_x, pfx_z (movement), plate_x, plate_z (location), release_speed, release_pos_x/y/z
- [ ] Pitch matching: match SAM3D pitches to Statcast by date + pitcher + inning + pitch_number_in_inning
- [ ] Correlation engine: Pearson/Spearman correlation between each biomechanical feature and each outcome metric
- [ ] Regression: which combination of mechanical features best predicts velocity? spin? movement?
- [ ] "Mechanics that matter" report: ranked list of features by outcome explanatory power
- [ ] Per-pitch-type: separate correlations for FB, SL, CB, CH
- [ ] Visualization data: scatter plots of mechanic X vs outcome Y with regression line

## Technical Approach

pybaseball library for Statcast data (free, well-maintained). Pitch matching is approximate — match by pitcher + game date + inning + pitch count context. Won't be perfect for broadcast video (camera cuts lose pitch order) but works well for bullpen sessions where you control the sequence.

Correlation analysis: Pearson for linear relationships, Spearman for monotonic. Multiple regression (Ridge or LASSO) for "best subset" of mechanics explaining an outcome. Feature importance from LASSO coefficients.

This is where the tool goes from "here are your mechanics" to "here's why your slider doesn't slide." That's the insight player dev people pay for.

## Files

| File | Purpose |
|------|---------|
| backend/app/data/statcast.py | StatcastFetcher, data matching |
| backend/app/analysis/correlation.py | Mechanics→outcome correlation engine |
| backend/tests/test_statcast.py | Statcast integration tests |

## Tasks

1. Implement Statcast data fetcher (pybaseball or CSV)
2. Build pitch matching engine (SAM3D ↔ Statcast)
3. Implement correlation analysis (per-feature, per-outcome)
4. Build "mechanics that matter" regression analysis
5. Generate correlation report with visualization data

## Dependencies

- Upstream: data-model
- Downstream: api-layer, ai-scouting-reports
