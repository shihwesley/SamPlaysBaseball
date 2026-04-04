"""Statcast data fetching and pitch matching.

Supports two fetch modes:
  1. pybaseball (live API) — requires internet; rate-limited.
  2. CSV file — pre-downloaded Statcast export.

Pitch matching uses: game_date + pitcher_id + inning + pitch_number.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from backend.app.models.pitch import PitchMetadata

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Statcast columns we care about
_KEEP_COLS = [
    "game_date",
    "pitcher",
    "inning",
    "pitch_number",
    "pitch_type",
    "release_speed",
    "release_spin_rate",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "events",
    "description",
    "woba_value",
    "estimated_woba_using_speedangle",
    "launch_speed",
    "launch_angle",
    "effective_speed",
    "release_extension",
    "spin_axis",
    "vx0",
    "vy0",
    "vz0",
    "ax",
    "ay",
    "az",
]


class StatcastFetcher:
    """Fetch and cache Statcast pitching data.

    Usage:
        fetcher = StatcastFetcher()
        df = fetcher.fetch_pitcher(pitcher_mlbam_id=543037, start="2023-04-01", end="2023-10-01")
        row = fetcher.match_pitch(df, metadata)
    """

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Fetching
    # ------------------------------------------------------------------

    def fetch_pitcher(
        self,
        pitcher_mlbam_id: int,
        start: str | date,
        end: str | date,
    ) -> pd.DataFrame:
        """Fetch Statcast pitching data for one pitcher over a date range.

        Tries cache first, then pybaseball.

        Args:
            pitcher_mlbam_id: MLB Advanced Media pitcher ID.
            start: start date string "YYYY-MM-DD" or date object.
            end: end date string "YYYY-MM-DD" or date object.

        Returns:
            DataFrame with Statcast columns. Empty if not found.
        """
        start_str = str(start) if isinstance(start, date) else start
        end_str = str(end) if isinstance(end, date) else end

        cache_path = self._cache_path(pitcher_mlbam_id, start_str, end_str)
        if cache_path and cache_path.exists():
            logger.info("Loading Statcast from cache: %s", cache_path)
            return pd.read_parquet(cache_path)

        df = self._fetch_from_pybaseball(pitcher_mlbam_id, start_str, end_str)

        if cache_path and not df.empty:
            df.to_parquet(cache_path, index=False)

        return df

    def load_csv(self, path: str | Path) -> pd.DataFrame:
        """Load Statcast data from a CSV file (e.g., Baseball Savant export).

        Args:
            path: path to CSV file.

        Returns:
            Cleaned DataFrame.
        """
        df = pd.read_csv(path, low_memory=False)
        return self._clean(df)

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def match_pitch(
        self,
        statcast_df: pd.DataFrame,
        metadata: PitchMetadata,
        tolerance_seconds: int = 0,
    ) -> pd.Series | None:
        """Match a PitchMetadata record to a Statcast row.

        Matching keys: game_date + pitcher (MLBAM ID) + inning + pitch_number.

        Args:
            statcast_df: DataFrame from fetch_pitcher or load_csv.
            metadata: PitchMetadata from SAM 3D analysis.
            tolerance_seconds: unused (reserved for future fuzzy date match).

        Returns:
            Matched row as pd.Series, or None if no match.
        """
        if statcast_df.empty:
            return None

        game_date = metadata.game_date.date() if isinstance(metadata.game_date, datetime) else metadata.game_date

        try:
            pitcher_id = int(metadata.pitcher_id)
        except (ValueError, TypeError):
            logger.warning("pitcher_id %r cannot be cast to int for Statcast match", metadata.pitcher_id)
            return None

        mask = (
            (pd.to_datetime(statcast_df["game_date"]).dt.date == game_date)
            & (statcast_df["pitcher"].astype(int) == pitcher_id)
            & (statcast_df["inning"].astype(int) == metadata.inning)
            & (statcast_df["pitch_number"].astype(int) == metadata.pitch_number)
        )
        matches = statcast_df[mask]
        if matches.empty:
            return None
        if len(matches) > 1:
            logger.warning("Multiple Statcast rows matched pitch %s — using first", metadata.pitch_id)
        return matches.iloc[0]

    def match_all(
        self,
        statcast_df: pd.DataFrame,
        metadatas: list[PitchMetadata],
    ) -> dict[str, pd.Series | None]:
        """Match a list of PitchMetadata records.

        Returns:
            Dict mapping pitch_id → matched row (or None).
        """
        return {m.pitch_id: self.match_pitch(statcast_df, m) for m in metadatas}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fetch_from_pybaseball(
        self,
        pitcher_mlbam_id: int,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        try:
            import pybaseball  # type: ignore[import]
        except ImportError:
            logger.error("pybaseball not installed. Run: pip install pybaseball")
            return pd.DataFrame()

        try:
            df = pybaseball.statcast_pitcher(start, end, pitcher_mlbam_id)
            return self._clean(df)
        except Exception as exc:
            logger.error("pybaseball fetch failed: %s", exc)
            return pd.DataFrame()

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Retain only known useful columns that exist in the DataFrame."""
        keep = [c for c in _KEEP_COLS if c in df.columns]
        return df[keep].copy()

    def _cache_path(
        self,
        pitcher_id: int,
        start: str,
        end: str,
    ) -> Path | None:
        if not self.cache_dir:
            return None
        fname = f"statcast_{pitcher_id}_{start}_{end}.parquet"
        return self.cache_dir / fname
