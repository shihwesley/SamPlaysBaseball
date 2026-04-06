"""MLB player search and game discovery.

Bridges natural-language player queries to MLBAM IDs, game logs,
and per-pitch Statcast data. Uses the MLB Stats API (no auth required).

Usage:
    searcher = PlayerSearch()

    # Find a player
    results = searcher.search("yamamoto")
    # [Player(id=808967, name="Yoshinobu Yamamoto", position="P", team="Los Angeles Dodgers")]

    # Get their pitching game log
    games = searcher.pitching_games(808967, season=2025)
    # [GameAppearance(game_pk=778563, date="2025-03-18", opponent="Chicago Cubs", ip=5.0, ...)]

    # Get per-pitch breakdown for a game
    pitches = searcher.game_pitches(game_pk=778563, pitcher_id=808967)
    # [PitchInfo(play_id="abc...", inning=1, pitch_type="FF", speed=96.2, ...)]
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

MLB_STATS_API = "https://statsapi.mlb.com/api/v1"
SAVANT_GAME_FEED = "https://baseballsavant.mlb.com/gf?game_pk={game_pk}"
SAVANT_STATCAST = (
    "https://baseballsavant.mlb.com/statcast_search/csv"
    "?all=true&player_type=pitcher"
    "&pitchers_lookup%5B%5D={pitcher_id}"
    "&game_date_gt={date}&game_date_lt={date}"
    "&type=details"
)

HEADERS = {
    "User-Agent": "SamPlaysBaseball/1.0",
    "Accept": "application/json",
}


def _parse_height_to_meters(height_str: str) -> float | None:
    """Parse MLB height string like '6\' 4\"' to meters."""
    if not height_str:
        return None
    try:
        clean = height_str.replace('"', '').strip()
        parts = clean.split("'")
        feet = int(parts[0].strip())
        inches = int(parts[1].strip()) if len(parts) > 1 and parts[1].strip() else 0
        return (feet * 12 + inches) * 0.0254
    except (ValueError, IndexError):
        return None


@dataclass
class Player:
    id: int
    name: str
    position: str
    team: str | None = None
    height_m: float | None = None
    weight_lbs: int | None = None


@dataclass
class GameAppearance:
    game_pk: int
    date: str
    opponent: str
    home_away: str  # "home" or "away"
    innings_pitched: float
    strikeouts: int
    earned_runs: int
    hits: int
    walks: int
    pitches_thrown: int | None = None


@dataclass
class PitchInfo:
    """Per-pitch data from the Baseball Savant game feed."""
    play_id: str
    inning: int
    at_bat_number: int
    pitch_type: str
    release_speed: float | None
    description: str
    batter_name: str
    events: str | None = None


def _fetch_json(url: str) -> dict:
    req = Request(url, headers=HEADERS)
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fetch_text(url: str) -> str:
    req = Request(url, headers=HEADERS)
    with urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


class PlayerSearch:
    """Search MLB players and discover their games."""

    def search(self, name: str, sport_id: int = 1) -> list[Player]:
        """Search for players by name. sport_id=1 is MLB.

        Returns matching players sorted by relevance.
        """
        url = f"{MLB_STATS_API}/people/search?names={name}&sportIds={sport_id}"
        data = _fetch_json(url)

        players = []
        for p in data.get("people", []):
            player = Player(
                id=p["id"],
                name=p.get("fullName", ""),
                position=p.get("primaryPosition", {}).get("abbreviation", "?"),
                team=p.get("currentTeam", {}).get("name"),
                height_m=_parse_height_to_meters(p.get("height", "")),
                weight_lbs=p.get("weight"),
            )
            players.append(player)

        return players

    def search_pitcher(self, name: str) -> list[Player]:
        """Search specifically for pitchers."""
        all_players = self.search(name)
        return [p for p in all_players if p.position == "P"]

    def pitching_games(
        self,
        pitcher_id: int,
        season: int = 2025,
    ) -> list[GameAppearance]:
        """Get all pitching appearances for a season."""
        url = (
            f"{MLB_STATS_API}/people/{pitcher_id}/stats"
            f"?stats=gameLog&group=pitching&season={season}"
        )
        data = _fetch_json(url)

        stats = data.get("stats", [])
        if not stats:
            return []

        games = []
        for split in stats[0].get("splits", []):
            s = split["stat"]
            opp = split.get("opponent", {}).get("name", "?")
            is_home = split.get("isHome", False)

            # Parse IP (can be "5.2" meaning 5 and 2/3)
            ip_str = s.get("inningsPitched", "0")
            try:
                ip = float(ip_str)
            except ValueError:
                ip = 0.0

            game = GameAppearance(
                game_pk=split.get("game", {}).get("gamePk", 0),
                date=split.get("date", ""),
                opponent=opp,
                home_away="home" if is_home else "away",
                innings_pitched=ip,
                strikeouts=int(s.get("strikeOuts", 0)),
                earned_runs=int(s.get("earnedRuns", 0)),
                hits=int(s.get("hits", 0)),
                walks=int(s.get("baseOnBalls", 0)),
                pitches_thrown=int(s.get("numberOfPitches", 0)) if "numberOfPitches" in s else None,
            )
            games.append(game)

        return games

    def game_pitches(
        self,
        game_pk: int,
        pitcher_id: int,
    ) -> list[PitchInfo]:
        """Get per-pitch data from Baseball Savant game feed."""
        url = SAVANT_GAME_FEED.format(game_pk=game_pk)
        data = _fetch_json(url)

        pitches = []
        for key in ["team_home", "team_away"]:
            for entry in data.get(key, []):
                if entry.get("pitcher") == pitcher_id:
                    pitch = PitchInfo(
                        play_id=entry["play_id"],
                        inning=entry.get("inning", 0),
                        at_bat_number=entry.get("ab_number", 0),
                        pitch_type=entry.get("pitch_type", "UN"),
                        release_speed=entry.get("release_speed"),
                        description=entry.get("description", ""),
                        batter_name=entry.get("batter_name", ""),
                        events=entry.get("event") or None,
                    )
                    pitches.append(pitch)

        return pitches

    def game_pitches_by_type(
        self,
        game_pk: int,
        pitcher_id: int,
        pitch_type: str,
    ) -> list[PitchInfo]:
        """Get all pitches of a specific type from a game.

        pitch_type: FF (4-seam), SI (sinker), SL (slider), CU (curve),
                    CH (changeup), FC (cutter), ST (sweeper), FS (splitter)
        """
        all_pitches = self.game_pitches(game_pk, pitcher_id)
        return [p for p in all_pitches if p.pitch_type == pitch_type]
