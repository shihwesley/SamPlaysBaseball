# Pitch Matching Engine Design

Status: designed, blocked on video pipeline output
Date: 2026-04-04

## Problem

Broadcast footage gives an *observed sequence* of pitches (with timestamps, rough velocity estimates). Statcast gives the *ground truth sequence* (exact velocity, spin, pitch type). These sequences may not align 1:1 — camera cuts skip pitches, replays duplicate them, commercial breaks create gaps.

No sub-second timestamps in Statcast. Alignment is ordinal (sequence position + velocity), not clock-based.

## Data Contracts

```python
@dataclass
class ObservedPitch:
    sequence_index: int              # ordinal position in video (0, 1, 2, ...)
    inning: int                      # detected or inferred from broadcast
    estimated_velocity: float | None # rough mph from optical flow
    timestamp_sec: float             # seconds from video start (for ordering)

class MatchTier(Enum):
    EXACT = "exact"                  # inning + pitch_number lined up perfectly
    SEQUENCE_ALIGNED = "aligned"     # DTW found a confident pairing
    NEAREST = "nearest"              # best-effort fallback
    UNMATCHED = "unmatched"          # couldn't match

@dataclass
class PitchMatch:
    observed: ObservedPitch
    statcast_row: pd.Series | None   # None if unmatched
    tier: MatchTier
    confidence: float                # 0.0–1.0
    distance: float | None           # alignment cost (lower = better)
```

## Tiered Algorithm

### Tier 1 — Exact Key Match
Existing `StatcastFetcher.match_pitch` logic. For observed pitches with known inning + pitch_number, look up the exact Statcast row. Cross-check velocity within ±5 mph tolerance. Confidence: 0.95 (with velocity check) or 0.85 (without).

### Tier 2 — DTW Sequence Alignment
Group remaining unmatched observed + Statcast pitches by inning. Run DTW per inning using cost function:

```
cost(observed, statcast) = |estimated_velocity - release_speed| / velocity_scale
```

velocity_scale = 20.0 (covers fastball-to-changeup spread). Confidence derived from cost: `max(0, 1.0 - pair_cost / cost_ceiling)`. Pairs above cost ceiling rejected to tier 3.

DTW chosen over Needleman-Wunsch: simpler, no gap penalty model needed, handles the common case (a few missing pitches). Swappable later if needed.

### Tier 3 — Nearest Neighbor
Remaining unmatched: closest unmatched Statcast pitch in same inning by velocity. Confidence capped at 0.4.

### Unmatched
Anything left: MatchTier.UNMATCHED, confidence 0.0.

Key constraint: DTW runs per inning (15–25 pitches), not across the full game.

## Module Structure

```
backend/app/data/
├── statcast.py              # existing — stays as-is
├── pitch_matcher.py         # NEW — PitchMatcher class
└── models/
    └── matching.py          # NEW — ObservedPitch, PitchMatch, MatchTier
```

PitchMatcher consumes StatcastFetcher output (DataFrame) but owns matching logic independently. Hand-rolled DTW in numpy (~30 lines), no external library needed for 15–25 element sequences.

```python
class PitchMatcher:
    def __init__(self, velocity_scale=20.0, dtw_cost_ceiling=1.5,
                 exact_velocity_tolerance=5.0):
    def match(self, observed: list[ObservedPitch],
              statcast_df: pd.DataFrame) -> list[PitchMatch]:
    def _exact_match(self, ...) -> ...:
    def _dtw_align(self, ...) -> ...:
    def _nearest_match(self, ...) -> ...:
```

## Testing Strategy

All synthetic data — no pybaseball mocking needed.

- **Tier 1:** perfect alignment, velocity cross-check failure, missing pitch_number fallthrough
- **Tier 2:** dropped pitches (20→15), replay duplication, velocity noise (±3 mph), mixed vs ambiguous speeds
- **Tier 3:** single orphan, no remaining candidates
- **Integration:** full game (7 innings, ~100 pitches), empty inputs, no double-assignment of Statcast rows

## Blocked On

Video pipeline producing real `ObservedPitch` sequences from broadcast footage. Build that first, then implement this engine against real data.
