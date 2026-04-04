"""Fetch per-pitch video clips from Baseball Savant.

Downloads individual pitch clips linked to Statcast data for a given
pitcher and game. Each clip is ~6-15 seconds of broadcast footage
showing one pitch delivery.

Usage:
    # Fetch all Ohtani pitches from WS Game 7
    python scripts/fetch_savant_clips.py --pitcher 660271 --game-pk 813024

    # Fetch by date range (downloads all games in range)
    python scripts/fetch_savant_clips.py --pitcher 660271 --start 2025-11-01 --end 2025-11-01

    # Specify output directory and video angle
    python scripts/fetch_savant_clips.py --pitcher 660271 --game-pk 813024 \
        --output-dir data/clips --angle HOME
"""

import argparse
import csv
import json
import logging
import re
import time
from dataclasses import dataclass, asdict
from io import StringIO
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SAVANT_SEARCH_URL = (
    "https://baseballsavant.mlb.com/statcast_search/csv"
    "?all=true&player_type=pitcher"
    "&pitchers_lookup%5B%5D={pitcher_id}"
    "&game_date_gt={start}&game_date_lt={end}"
    "&type=details"
)

GAME_FEED_URL = "https://baseballsavant.mlb.com/gf?game_pk={game_pk}"

SPORTY_VIDEO_URL = (
    "https://baseballsavant.mlb.com/sporty-videos"
    "?playId={play_id}&videoType={angle}"
)

# Headers that avoid Cloudflare blocks on the CDN
DOWNLOAD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://baseballsavant.mlb.com/",
    "Origin": "https://www.mlb.com",
}


@dataclass
class PitchClip:
    """One pitch with Statcast data + video path."""

    play_id: str
    game_pk: int
    game_date: str
    pitcher_id: int
    pitcher_name: str
    batter_name: str
    inning: int
    at_bat_number: int
    pitch_number_in_ab: int  # pitch number within at-bat
    pitch_type: str
    release_speed: float | None
    description: str  # ball, called_strike, swinging_strike, hit_into_play, etc.
    events: str | None  # strikeout, home_run, etc. (only on last pitch of AB)
    video_url: str
    video_path: str | None = None  # set after download


def _fetch_url(url: str, headers: dict | None = None) -> str:
    """Fetch URL content as string."""
    req = Request(url, headers=headers or DOWNLOAD_HEADERS)
    with urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def _download_clip(sporty_url: str, dest: Path) -> bool:
    """Download clip via yt-dlp. Returns True on success."""
    import subprocess

    result = subprocess.run(
        ["yt-dlp", "-q", "--no-warnings", "-o", str(dest), sporty_url],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0 or not dest.exists():
        logger.warning("yt-dlp failed: %s", result.stderr.strip())
        return False
    # Validate it's actually video, not an error page
    if dest.stat().st_size < 10_000:
        content = dest.read_bytes()[:100]
        if b"<!DOCTYPE" in content or content.lstrip().startswith(b"<"):
            logger.warning("Got HTML instead of video, removing")
            dest.unlink()
            return False
    return True


def get_game_pks(pitcher_id: int, start: str, end: str) -> list[int]:
    """Find game_pk values for a pitcher in a date range via Statcast search."""
    url = SAVANT_SEARCH_URL.format(pitcher_id=pitcher_id, start=start, end=end)
    text = _fetch_url(url)
    reader = csv.DictReader(StringIO(text))
    game_pks = set()
    for row in reader:
        gk = row.get("game_pk")
        if gk:
            game_pks.add(int(gk))
    return sorted(game_pks)


def get_pitch_play_ids(game_pk: int, pitcher_id: int) -> list[dict]:
    """Get play_id and pitch metadata from the game feed endpoint.

    Returns list of dicts with play_id, inning, pitch_type, batter_name, etc.
    """
    url = GAME_FEED_URL.format(game_pk=game_pk)
    text = _fetch_url(url)
    data = json.loads(text)

    pitches = []
    for key in ["team_home", "team_away"]:
        for entry in data.get(key, []):
            if entry.get("pitcher") == pitcher_id:
                pitches.append(entry)

    return pitches


def extract_mp4_url(sporty_page_html: str) -> str | None:
    """Pull the .mp4 URL from a sporty-videos page."""
    match = re.search(r'https://[^"\']+\.mp4', sporty_page_html)
    return match.group(0) if match else None


def fetch_game_clips(
    game_pk: int,
    pitcher_id: int,
    angle: str = "AWAY",
    output_dir: Path = Path("data/clips"),
    delay: float = 1.0,
    pitch_types: list[str] | None = None,
) -> list[PitchClip]:
    """Fetch all pitch clips for a pitcher in one game.

    Args:
        game_pk: MLB game primary key.
        pitcher_id: MLBAM pitcher ID.
        angle: HOME or AWAY broadcast feed.
        output_dir: where to save clips.
        delay: seconds between downloads (be nice to MLB servers).
        pitch_types: filter to these types only (e.g., ["FF", "SI"]). None = all.

    Returns:
        List of PitchClip with video_path set for successful downloads.
    """
    logger.info("Fetching game feed for game_pk=%d...", game_pk)
    pitch_entries = get_pitch_play_ids(game_pk, pitcher_id)
    logger.info("Found %d pitches for pitcher %d", len(pitch_entries), pitcher_id)

    if pitch_types:
        pitch_entries = [e for e in pitch_entries if e.get("pitch_type") in pitch_types]
        logger.info("Filtered to %d pitches (types: %s)", len(pitch_entries), ", ".join(pitch_types))

    if not pitch_entries:
        return []

    game_dir = output_dir / str(game_pk)
    game_dir.mkdir(parents=True, exist_ok=True)

    clips = []
    for i, entry in enumerate(pitch_entries):
        play_id = entry["play_id"]
        inning = entry.get("inning", 0)
        ab_number = entry.get("ab_number", 0)
        pitch_type = entry.get("pitch_type", "UN")
        description = entry.get("description", "")
        events = entry.get("event", "")
        batter_name = entry.get("batter_name", "unknown")
        pitcher_name = entry.get("pitcher_name", "unknown")

        # Count pitch number within at-bat
        ab_pitches = [e for e in pitch_entries[:i+1] if e.get("ab_number") == ab_number]
        pitch_num_in_ab = len(ab_pitches)

        # Get video URL from sporty-videos page
        sporty_url = SPORTY_VIDEO_URL.format(play_id=play_id, angle=angle)
        try:
            page_html = _fetch_url(sporty_url)
            mp4_url = extract_mp4_url(page_html)
        except Exception as e:
            logger.warning("Failed to get video page for pitch %d: %s", i + 1, e)
            mp4_url = None

        if not mp4_url:
            logger.warning("No MP4 URL found for pitch %d (play_id=%s)", i + 1, play_id)
            mp4_url = sporty_url  # fallback for yt-dlp

        clip = PitchClip(
            play_id=play_id,
            game_pk=game_pk,
            game_date=entry.get("game_date", ""),
            pitcher_id=pitcher_id,
            pitcher_name=pitcher_name,
            batter_name=batter_name,
            inning=inning,
            at_bat_number=ab_number,
            pitch_number_in_ab=pitch_num_in_ab,
            pitch_type=pitch_type,
            release_speed=entry.get("release_speed"),
            description=description,
            events=events or None,
            video_url=mp4_url,
        )

        # Download clip
        filename = f"inn{inning}_ab{ab_number}_p{pitch_num_in_ab}_{pitch_type}_{play_id[:8]}.mp4"
        dest = game_dir / filename

        if dest.exists() and dest.stat().st_size > 10_000:
            logger.info("  [%d/%d] Already downloaded: %s", i + 1, len(pitch_entries), filename)
            clip.video_path = str(dest)
        else:
            if dest.exists():
                dest.unlink()  # remove bad file from previous run
            logger.info(
                "  [%d/%d] Inn %d | %s %s → %s | %s",
                i + 1, len(pitch_entries),
                inning, pitch_type,
                entry.get("release_speed", "?"),
                batter_name, description,
            )
            if _download_clip(sporty_url, dest):
                clip.video_path = str(dest)
            else:
                logger.warning("    No video available for pitch %d", i + 1)

            if delay > 0 and i < len(pitch_entries) - 1:
                time.sleep(delay)

        clips.append(clip)

    # Save manifest
    manifest_path = game_dir / "manifest.json"
    manifest = {
        "game_pk": game_pk,
        "pitcher_id": pitcher_id,
        "pitcher_name": clips[0].pitcher_name if clips else "",
        "angle": angle,
        "total_pitches": len(clips),
        "downloaded": sum(1 for c in clips if c.video_path),
        "pitches": [asdict(c) for c in clips],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Manifest saved: %s", manifest_path)

    downloaded = sum(1 for c in clips if c.video_path)
    logger.info("Done: %d/%d clips downloaded to %s", downloaded, len(clips), game_dir)

    return clips


def resolve_pitcher(name_or_id: str) -> tuple[int, str]:
    """Resolve a pitcher name or ID to (mlbam_id, display_name).

    Accepts:
        "660271"           → (660271, "Shohei Ohtani")
        "ohtani"           → (660271, "Shohei Ohtani")
        "yamamoto"         → (808967, "Yoshinobu Yamamoto")
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from backend.app.data.player_search import PlayerSearch

    # If it's a number, look up the name
    try:
        pid = int(name_or_id)
        ps = PlayerSearch()
        # Reverse lookup not available, just return the ID
        return pid, f"Pitcher #{pid}"
    except ValueError:
        pass

    # Search by name
    ps = PlayerSearch()
    pitchers = ps.search_pitcher(name_or_id)
    if not pitchers:
        # Try all positions
        all_players = ps.search(name_or_id)
        if all_players:
            logger.error("Found players but none are pitchers: %s",
                         ", ".join(f"{p.name} ({p.position})" for p in all_players))
        else:
            logger.error("No players found matching '%s'", name_or_id)
        sys.exit(1)

    if len(pitchers) == 1:
        p = pitchers[0]
        logger.info("Found pitcher: %s (ID: %d)", p.name, p.id)
        return p.id, p.name

    # Multiple matches — show options
    logger.info("Multiple pitchers found:")
    for i, p in enumerate(pitchers):
        logger.info("  [%d] %s (ID: %d, Team: %s)", i + 1, p.name, p.id, p.team or "?")
    choice = input("Pick a number (or press Enter for #1): ").strip()
    idx = int(choice) - 1 if choice else 0
    p = pitchers[idx]
    return p.id, p.name


def main():
    parser = argparse.ArgumentParser(
        description="Fetch per-pitch video clips from Baseball Savant"
    )
    parser.add_argument("--pitcher", type=str, required=True,
                        help="Pitcher name or MLBAM ID (e.g., 'ohtani' or 660271)")
    parser.add_argument("--game-pk", type=int, default=None, help="Specific game_pk to fetch")
    parser.add_argument("--season", type=int, default=None,
                        help="Show game list for this season and let user pick")
    parser.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD (used with --end)")
    parser.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--output-dir", type=str, default="data/clips", help="Output directory for clips")
    parser.add_argument("--angle", choices=["HOME", "AWAY"], default="AWAY", help="Broadcast angle")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between downloads")
    parser.add_argument("--pitch-type", type=str, default=None,
                        help="Filter to pitch type(s), comma-separated (e.g., 'FF' or 'FF,SI,FC')")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    pitcher_id, pitcher_name = resolve_pitcher(args.pitcher)
    logger.info("Pitcher: %s (ID: %d)", pitcher_name, pitcher_id)

    if args.game_pk:
        game_pks = [args.game_pk]
    elif args.season:
        # Show game log and let user pick
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from backend.app.data.player_search import PlayerSearch
        ps = PlayerSearch()
        games = ps.pitching_games(pitcher_id, season=args.season)
        if not games:
            logger.error("No pitching appearances in %d", args.season)
            sys.exit(1)
        logger.info("%s — %d games in %d:", pitcher_name, len(games), args.season)
        for i, g in enumerate(games):
            logger.info("  [%2d] %s vs %-25s IP=%s K=%d ER=%d (GP=%d)",
                        i + 1, g.date, g.opponent, g.innings_pitched, g.strikeouts, g.earned_runs, g.game_pk)
        choice = input("\nPick game number(s), comma-separated (e.g., '1' or '1,2,3'): ").strip()
        indices = [int(x.strip()) - 1 for x in choice.split(",")]
        game_pks = [games[i].game_pk for i in indices]
        logger.info("Selected games: %s", game_pks)
    elif args.start and args.end:
        logger.info("Finding games for pitcher %d from %s to %s...", pitcher_id, args.start, args.end)
        game_pks = get_game_pks(pitcher_id, args.start, args.end)
        logger.info("Found %d games: %s", len(game_pks), game_pks)
    else:
        parser.error("Provide --game-pk, --season, or both --start and --end")

    pitch_types = [t.strip().upper() for t in args.pitch_type.split(",")] if args.pitch_type else None

    all_clips = []
    for gk in game_pks:
        clips = fetch_game_clips(
            game_pk=gk,
            pitcher_id=pitcher_id,
            angle=args.angle,
            output_dir=output_dir,
            delay=args.delay,
            pitch_types=pitch_types,
        )
        all_clips.extend(clips)

    total = len(all_clips)
    downloaded = sum(1 for c in all_clips if c.video_path)
    logger.info("\nTotal: %d pitches, %d downloaded across %d games", total, downloaded, len(game_pks))


if __name__ == "__main__":
    main()
