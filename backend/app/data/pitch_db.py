"""SQLite pitch database + .npz mesh storage.

Stores pitch metadata in SQLite (queryable) and 3D mesh/skeleton
arrays in .npz files (efficient for large numpy arrays).

Usage:
    db = PitchDB("data/pitches.db", mesh_dir="data/meshes")
    db.insert_pitch(pitch_record, mesh_data)
    results = db.query("SELECT * FROM pitches WHERE pitch_type = 'FF' AND release_speed > 95")
    mesh = db.load_mesh("02ec65f0-9054-3c9c-a72a-aaa3c610f0c9")
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS pitches (
    play_id TEXT PRIMARY KEY,
    game_pk INTEGER NOT NULL,
    game_date TEXT NOT NULL,
    pitcher_id INTEGER NOT NULL,
    pitcher_name TEXT,
    batter_name TEXT,
    inning INTEGER NOT NULL,
    at_bat_number INTEGER,
    pitch_number_in_ab INTEGER,
    pitch_type TEXT,
    release_speed REAL,
    release_spin_rate REAL,
    spin_axis REAL,
    pfx_x REAL,
    pfx_z REAL,
    plate_x REAL,
    plate_z REAL,
    effective_speed REAL,
    release_extension REAL,
    description TEXT,
    events TEXT,
    woba_value REAL,
    estimated_woba REAL,
    launch_speed REAL,
    launch_angle REAL,
    video_path TEXT,
    mesh_path TEXT,
    num_frames INTEGER,
    inference_time_ms REAL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_pitches_game ON pitches(game_pk);
CREATE INDEX IF NOT EXISTS idx_pitches_pitcher ON pitches(pitcher_id);
CREATE INDEX IF NOT EXISTS idx_pitches_type ON pitches(pitch_type);
CREATE INDEX IF NOT EXISTS idx_pitches_inning ON pitches(game_pk, inning);
"""


@dataclass
class PitchRecord:
    """Flat representation of one pitch for database storage."""

    play_id: str
    game_pk: int
    game_date: str
    pitcher_id: int
    pitcher_name: str = ""
    batter_name: str = ""
    inning: int = 0
    at_bat_number: int | None = None
    pitch_number_in_ab: int | None = None
    pitch_type: str | None = None
    release_speed: float | None = None
    release_spin_rate: float | None = None
    spin_axis: float | None = None
    pfx_x: float | None = None          # horizontal movement (inches)
    pfx_z: float | None = None          # vertical movement (inches)
    plate_x: float | None = None        # horizontal location at plate (feet)
    plate_z: float | None = None        # vertical location at plate (feet)
    effective_speed: float | None = None
    release_extension: float | None = None
    description: str | None = None
    events: str | None = None
    woba_value: float | None = None
    estimated_woba: float | None = None
    launch_speed: float | None = None
    launch_angle: float | None = None
    video_path: str | None = None
    mesh_path: str | None = None
    num_frames: int | None = None
    inference_time_ms: float | None = None


@dataclass
class MeshData:
    """3D mesh + skeleton arrays for one pitch clip."""

    vertices: np.ndarray       # (T, N, 3) mesh vertices per frame
    joints_mhr70: np.ndarray   # (T, 70, 3) skeleton joints per frame
    pose_params: np.ndarray    # (T, 136) SMPL pose per frame
    shape_params: np.ndarray   # (45,) body shape (constant across frames)
    cam_t: np.ndarray          # (T, 3) camera translation per frame
    focal_length: float        # for re-rendering


class PitchDB:
    """SQLite pitch database with .npz mesh storage."""

    def __init__(
        self,
        db_path: str | Path = "data/pitches.db",
        mesh_dir: str | Path = "data/meshes",
    ) -> None:
        self.db_path = Path(db_path)
        self.mesh_dir = Path(mesh_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.mesh_dir.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> PitchDB:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert_pitch(
        self,
        record: PitchRecord,
        mesh: MeshData | None = None,
    ) -> None:
        """Insert a pitch record, optionally with mesh data."""
        if mesh is not None:
            mesh_path = self._save_mesh(record.play_id, record.game_pk, mesh)
            record.mesh_path = str(mesh_path)
            record.num_frames = mesh.vertices.shape[0]

        cols = [
            "play_id", "game_pk", "game_date", "pitcher_id", "pitcher_name",
            "batter_name", "inning", "at_bat_number", "pitch_number_in_ab",
            "pitch_type", "release_speed", "release_spin_rate", "spin_axis",
            "pfx_x", "pfx_z", "plate_x", "plate_z",
            "effective_speed", "release_extension",
            "description", "events", "woba_value", "estimated_woba",
            "launch_speed", "launch_angle",
            "video_path", "mesh_path", "num_frames", "inference_time_ms",
        ]
        vals = [getattr(record, c) for c in cols]
        placeholders = ", ".join(["?"] * len(cols))
        col_names = ", ".join(cols)

        self._conn.execute(
            f"INSERT OR REPLACE INTO pitches ({col_names}) VALUES ({placeholders})",
            vals,
        )
        self._conn.commit()

    def insert_from_manifest(self, manifest_path: str | Path) -> int:
        """Bulk-insert pitches from a fetch_savant_clips manifest.json.

        Returns number of pitches inserted.
        """
        manifest = json.loads(Path(manifest_path).read_text())
        count = 0
        for p in manifest["pitches"]:
            record = PitchRecord(
                play_id=p["play_id"],
                game_pk=p["game_pk"],
                game_date=p.get("game_date", ""),
                pitcher_id=p["pitcher_id"],
                pitcher_name=p.get("pitcher_name", ""),
                batter_name=p.get("batter_name", ""),
                inning=p.get("inning", 0),
                at_bat_number=p.get("at_bat_number"),
                pitch_number_in_ab=p.get("pitch_number_in_ab"),
                pitch_type=p.get("pitch_type"),
                release_speed=p.get("release_speed"),
                description=p.get("description"),
                events=p.get("events"),
                video_path=p.get("video_path"),
            )
            self.insert_pitch(record)
            count += 1
        return count

    def update_mesh(
        self,
        play_id: str,
        mesh: MeshData,
        inference_time_ms: float | None = None,
    ) -> None:
        """Attach mesh data to an existing pitch record."""
        row = self._conn.execute(
            "SELECT game_pk FROM pitches WHERE play_id = ?", (play_id,)
        ).fetchone()
        if not row:
            raise ValueError(f"No pitch found with play_id={play_id}")

        mesh_path = self._save_mesh(play_id, row["game_pk"], mesh)
        num_frames = mesh.vertices.shape[0]

        self._conn.execute(
            "UPDATE pitches SET mesh_path = ?, num_frames = ?, inference_time_ms = ? WHERE play_id = ?",
            (str(mesh_path), num_frames, inference_time_ms, play_id),
        )
        self._conn.commit()

    def enrich_from_statcast(
        self, game_pk: int, pitcher_id: int, game_date: str | None = None,
    ) -> int:
        """Fetch Statcast CSV data and merge into existing pitch records.

        Matches pitches by game_pk + pitcher_id + inning + pitch order.
        Returns number of pitches enriched.
        """
        import csv
        from io import StringIO
        from urllib.request import Request, urlopen

        # Resolve game date
        if not game_date:
            row = self._conn.execute(
                "SELECT game_date FROM pitches WHERE game_pk = ? AND pitcher_id = ? AND game_date != '' LIMIT 1",
                (game_pk, pitcher_id),
            ).fetchone()
            game_date = row["game_date"] if row else None

        if not game_date:
            # Look up from MLB Stats API
            try:
                api_url = f"https://statsapi.mlb.com/api/v1/schedule?gamePk={game_pk}"
                req = Request(api_url, headers={"User-Agent": "SamPlaysBaseball/1.0"})
                with urlopen(req, timeout=15) as resp:
                    import json as _json
                    sched = _json.loads(resp.read().decode("utf-8"))
                game_date = sched["dates"][0]["games"][0]["officialDate"]
                logger.info("Resolved game_date from MLB API: %s", game_date)
            except Exception as e:
                logger.warning("Cannot resolve game_date for game_pk=%d: %s", game_pk, e)
                return 0

        url = (
            f"https://baseballsavant.mlb.com/statcast_search/csv"
            f"?all=true&player_type=pitcher"
            f"&pitchers_lookup%5B%5D={pitcher_id}"
            f"&game_date_gt={game_date}&game_date_lt={game_date}"
            f"&type=details"
        )
        headers = {"User-Agent": "SamPlaysBaseball/1.0"}
        req = Request(url, headers=headers)
        with urlopen(req, timeout=30) as resp:
            text = resp.read().decode("utf-8")

        reader = csv.DictReader(StringIO(text))
        statcast_rows = [r for r in reader if r.get("game_pk") == str(game_pk)]

        if not statcast_rows:
            logger.warning("No Statcast data found for game_pk=%d on %s", game_pk, game_date)
            return 0

        # Get DB pitches in order
        db_pitches = self.query(
            "SELECT play_id, inning, at_bat_number, pitch_number_in_ab, pitch_type "
            "FROM pitches WHERE game_pk = ? AND pitcher_id = ? "
            "ORDER BY inning, at_bat_number, pitch_number_in_ab",
            (game_pk, pitcher_id),
        )

        # Sort Statcast rows the same way
        statcast_rows.sort(key=lambda r: (
            int(r.get("inning", 0)),
            int(r.get("at_bat_number", 0)),
            int(r.get("pitch_number", 0)),
        ))

        def _float_or_none(val):
            if val is None or val == "" or val == "null":
                return None
            try:
                return float(val)
            except (ValueError, TypeError):
                return None

        # Match by position (both sorted by inning/AB/pitch order)
        enriched = 0
        for db_pitch, sc_row in zip(db_pitches, statcast_rows):
            play_id = db_pitch["play_id"]
            self._conn.execute(
                """UPDATE pitches SET
                    release_speed = COALESCE(?, release_speed),
                    release_spin_rate = COALESCE(?, release_spin_rate),
                    spin_axis = COALESCE(?, spin_axis),
                    pfx_x = COALESCE(?, pfx_x),
                    pfx_z = COALESCE(?, pfx_z),
                    plate_x = COALESCE(?, plate_x),
                    plate_z = COALESCE(?, plate_z),
                    effective_speed = COALESCE(?, effective_speed),
                    release_extension = COALESCE(?, release_extension),
                    woba_value = COALESCE(?, woba_value),
                    estimated_woba = COALESCE(?, estimated_woba),
                    launch_speed = COALESCE(?, launch_speed),
                    launch_angle = COALESCE(?, launch_angle),
                    game_date = COALESCE(NULLIF(game_date, ''), ?)
                WHERE play_id = ?""",
                (
                    _float_or_none(sc_row.get("release_speed")),
                    _float_or_none(sc_row.get("release_spin_rate")),
                    _float_or_none(sc_row.get("spin_axis")),
                    _float_or_none(sc_row.get("pfx_x")),
                    _float_or_none(sc_row.get("pfx_z")),
                    _float_or_none(sc_row.get("plate_x")),
                    _float_or_none(sc_row.get("plate_z")),
                    _float_or_none(sc_row.get("effective_speed")),
                    _float_or_none(sc_row.get("release_extension")),
                    _float_or_none(sc_row.get("woba_value")),
                    _float_or_none(sc_row.get("estimated_woba_using_speedangle")),
                    _float_or_none(sc_row.get("launch_speed")),
                    _float_or_none(sc_row.get("launch_angle")),
                    sc_row.get("game_date", ""),
                    play_id,
                ),
            )
            enriched += 1

        self._conn.commit()
        logger.info("Enriched %d/%d pitches with Statcast data", enriched, len(db_pitches))
        return enriched

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        """Run arbitrary SQL and return list of dicts."""
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_pitch(self, play_id: str) -> dict | None:
        """Get a single pitch record by play_id."""
        row = self._conn.execute(
            "SELECT * FROM pitches WHERE play_id = ?", (play_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_game_pitches(self, game_pk: int) -> list[dict]:
        """Get all pitches for a game, ordered by inning and at-bat."""
        return self.query(
            "SELECT * FROM pitches WHERE game_pk = ? ORDER BY inning, at_bat_number, pitch_number_in_ab",
            (game_pk,),
        )

    def get_by_type(self, pitch_type: str, pitcher_id: int | None = None) -> list[dict]:
        """Get all pitches of a given type, optionally filtered by pitcher."""
        if pitcher_id:
            return self.query(
                "SELECT * FROM pitches WHERE pitch_type = ? AND pitcher_id = ?",
                (pitch_type, pitcher_id),
            )
        return self.query(
            "SELECT * FROM pitches WHERE pitch_type = ?", (pitch_type,)
        )

    def summary(self) -> dict:
        """Quick database summary."""
        total = self._conn.execute("SELECT COUNT(*) FROM pitches").fetchone()[0]
        with_mesh = self._conn.execute("SELECT COUNT(*) FROM pitches WHERE mesh_path IS NOT NULL").fetchone()[0]
        games = self._conn.execute("SELECT COUNT(DISTINCT game_pk) FROM pitches").fetchone()[0]
        pitchers = self._conn.execute("SELECT COUNT(DISTINCT pitcher_id) FROM pitches").fetchone()[0]

        types = self.query(
            "SELECT pitch_type, COUNT(*) as count FROM pitches GROUP BY pitch_type ORDER BY count DESC"
        )
        return {
            "total_pitches": total,
            "with_mesh": with_mesh,
            "games": games,
            "pitchers": pitchers,
            "pitch_types": {r["pitch_type"]: r["count"] for r in types},
        }

    # ------------------------------------------------------------------
    # Mesh I/O
    # ------------------------------------------------------------------

    def load_mesh(self, play_id: str) -> MeshData | None:
        """Load mesh data for a pitch."""
        row = self._conn.execute(
            "SELECT mesh_path FROM pitches WHERE play_id = ?", (play_id,)
        ).fetchone()
        if not row or not row["mesh_path"]:
            return None

        path = Path(row["mesh_path"])
        if not path.exists():
            logger.warning("Mesh file missing: %s", path)
            return None

        data = np.load(path)
        return MeshData(
            vertices=data["vertices"],
            joints_mhr70=data["joints_mhr70"],
            pose_params=data["pose_params"],
            shape_params=data["shape_params"],
            cam_t=data["cam_t"],
            focal_length=float(data["focal_length"]),
        )

    def _save_mesh(self, play_id: str, game_pk: int, mesh: MeshData) -> Path:
        """Save mesh arrays to .npz, return path."""
        game_dir = self.mesh_dir / str(game_pk)
        game_dir.mkdir(parents=True, exist_ok=True)

        # Use first 8 chars of play_id for filename
        short_id = play_id[:8]
        path = game_dir / f"{short_id}.npz"

        np.savez_compressed(
            path,
            vertices=mesh.vertices,
            joints_mhr70=mesh.joints_mhr70,
            pose_params=mesh.pose_params,
            shape_params=mesh.shape_params,
            cam_t=mesh.cam_t,
            focal_length=np.array(mesh.focal_length),
        )
        return path
