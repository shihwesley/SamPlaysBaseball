"""SQLite + Parquet storage layer.

SQLite stores relational data: pitch metadata, pitcher info, analysis result summaries.
Parquet stores numpy arrays: joint positions, pose params.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .analysis import AnalysisResult
from .baseline import PitcherBaseline
from .pitch import PitchData, PitchMetadata


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS pitchers (
    pitcher_id TEXT PRIMARY KEY,
    name TEXT,
    handedness TEXT,
    baseline_json TEXT
);

CREATE TABLE IF NOT EXISTS pitches (
    pitch_id TEXT PRIMARY KEY,
    pitcher_id TEXT NOT NULL,
    game_date TEXT NOT NULL,
    inning INTEGER,
    pitch_number INTEGER,
    pitch_type TEXT,
    velocity_mph REAL,
    spin_rate_rpm REAL,
    plate_x REAL,
    plate_z REAL,
    result TEXT,
    video_path TEXT,
    frame_start INTEGER,
    frame_end INTEGER,
    num_frames INTEGER,
    parquet_path TEXT
);

CREATE TABLE IF NOT EXISTS analysis_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pitch_id TEXT NOT NULL,
    pitcher_id TEXT NOT NULL,
    module TEXT NOT NULL,
    result_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pitches_pitcher ON pitches(pitcher_id);
CREATE INDEX IF NOT EXISTS idx_analysis_pitch ON analysis_results(pitch_id);
CREATE INDEX IF NOT EXISTS idx_analysis_module ON analysis_results(module);
"""


class StorageLayer:
    """Handles all persistence for the baseball analyzer."""

    def __init__(self, db_path: str | Path, parquet_dir: str | Path):
        self.db_path = Path(db_path)
        self.parquet_dir = Path(parquet_dir)
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # DB init
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA_SQL)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Parquet helpers
    # ------------------------------------------------------------------

    def _parquet_path(self, pitch_id: str) -> Path:
        return self.parquet_dir / f"{pitch_id}.parquet"

    def _write_parquet(self, pitch: PitchData) -> Path:
        """Write numpy arrays for a pitch to parquet. Returns path."""
        joints = pitch.joints_array()  # (T, 127, 3)
        pose = pitch.pose_params_array()  # (T, 136)
        shape = pitch.shape_params_array()  # (45,)
        T = joints.shape[0]

        arrays: dict[str, pa.Array] = {}
        # Store joints as T rows of 127*3 floats
        arrays["joints"] = pa.array(joints.reshape(T, -1).tolist(), type=pa.list_(pa.float32()))
        arrays["pose_params"] = pa.array(pose.tolist(), type=pa.list_(pa.float32()))

        if pitch.joints_mhr70 is not None:
            j70 = pitch.joints_mhr70_array()  # (T, 70, 3)
            arrays["joints_mhr70"] = pa.array(
                j70.reshape(T, -1).tolist(), type=pa.list_(pa.float32())
            )

        table = pa.table(arrays)
        # Shape params: store as metadata on the table schema
        meta = {
            b"shape_params": json.dumps(shape.tolist()).encode(),
            b"joints_shape": json.dumps(list(joints.shape)).encode(),
            b"pose_shape": json.dumps(list(pose.shape)).encode(),
        }
        table = table.replace_schema_metadata(meta)

        path = self._parquet_path(pitch.metadata.pitch_id)
        pq.write_table(table, path)
        return path

    def _read_parquet(self, pitch_id: str) -> dict:
        """Read parquet for pitch_id. Returns dict of numpy arrays."""
        path = self._parquet_path(pitch_id)
        table = pq.read_table(path)
        meta = table.schema.metadata or {}

        joints_shape = json.loads(meta[b"joints_shape"])
        pose_shape = json.loads(meta[b"pose_shape"])
        shape_params = np.array(json.loads(meta[b"shape_params"]), dtype=np.float32)

        joints_flat = np.array(table["joints"].to_pylist(), dtype=np.float32)
        joints = joints_flat.reshape(joints_shape)

        pose_flat = np.array(table["pose_params"].to_pylist(), dtype=np.float32)
        pose = pose_flat.reshape(pose_shape)

        result = {
            "joints": joints,
            "pose_params": pose,
            "shape_params": shape_params,
        }

        if "joints_mhr70" in table.column_names:
            T = joints.shape[0]
            j70_flat = np.array(table["joints_mhr70"].to_pylist(), dtype=np.float32)
            result["joints_mhr70"] = j70_flat.reshape(T, 70, 3)

        return result

    # ------------------------------------------------------------------
    # PitchData CRUD
    # ------------------------------------------------------------------

    def save_pitch(self, pitch: PitchData) -> None:
        parquet_path = self._write_parquet(pitch)
        m = pitch.metadata
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO pitches
                (pitch_id, pitcher_id, game_date, inning, pitch_number, pitch_type,
                 velocity_mph, spin_rate_rpm, plate_x, plate_z, result,
                 video_path, frame_start, frame_end, num_frames, parquet_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    m.pitch_id,
                    m.pitcher_id,
                    m.game_date.isoformat(),
                    m.inning,
                    m.pitch_number,
                    m.pitch_type,
                    m.velocity_mph,
                    m.spin_rate_rpm,
                    m.plate_x,
                    m.plate_z,
                    m.result,
                    m.video_path,
                    m.frame_start,
                    m.frame_end,
                    pitch.num_frames,
                    str(parquet_path),
                ),
            )

    def load_pitch(self, pitch_id: str) -> PitchData | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM pitches WHERE pitch_id = ?", (pitch_id,)
            ).fetchone()
        if row is None:
            return None

        from datetime import datetime

        meta = PitchMetadata(
            pitch_id=row["pitch_id"],
            pitcher_id=row["pitcher_id"],
            game_date=datetime.fromisoformat(row["game_date"]),
            inning=row["inning"],
            pitch_number=row["pitch_number"],
            pitch_type=row["pitch_type"],
            velocity_mph=row["velocity_mph"],
            spin_rate_rpm=row["spin_rate_rpm"],
            plate_x=row["plate_x"],
            plate_z=row["plate_z"],
            result=row["result"],
            video_path=row["video_path"],
            frame_start=row["frame_start"],
            frame_end=row["frame_end"],
        )
        arrays = self._read_parquet(pitch_id)
        return PitchData.from_numpy(
            metadata=meta,
            joints=arrays["joints"],
            pose_params=arrays["pose_params"],
            shape_params=arrays["shape_params"],
            joints_mhr70=arrays.get("joints_mhr70"),
        )

    def list_pitch_ids(self, pitcher_id: str | None = None) -> list[str]:
        with self._conn() as conn:
            if pitcher_id:
                rows = conn.execute(
                    "SELECT pitch_id FROM pitches WHERE pitcher_id = ? ORDER BY game_date, pitch_number",
                    (pitcher_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT pitch_id FROM pitches ORDER BY game_date, pitch_number"
                ).fetchall()
        return [r["pitch_id"] for r in rows]

    # ------------------------------------------------------------------
    # PitcherBaseline CRUD
    # ------------------------------------------------------------------

    def save_baseline(self, baseline: PitcherBaseline) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO pitchers (pitcher_id, name, handedness, baseline_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    baseline.pitcher_id,
                    baseline.pitcher_name,
                    baseline.handedness,
                    baseline.model_dump_json(),
                ),
            )

    def load_baseline(self, pitcher_id: str) -> PitcherBaseline | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT baseline_json FROM pitchers WHERE pitcher_id = ?", (pitcher_id,)
            ).fetchone()
        if row is None:
            return None
        return PitcherBaseline.model_validate_json(row["baseline_json"])

    # ------------------------------------------------------------------
    # AnalysisResult CRUD
    # ------------------------------------------------------------------

    def save_analysis(self, result: AnalysisResult) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO analysis_results (pitch_id, pitcher_id, module, result_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    result.pitch_id,
                    result.pitcher_id,
                    result.module,
                    result.model_dump_json(),
                ),
            )

    def load_analysis(self, pitch_id: str, module: str | None = None) -> list[dict]:
        with self._conn() as conn:
            if module:
                rows = conn.execute(
                    "SELECT result_json FROM analysis_results WHERE pitch_id = ? AND module = ?",
                    (pitch_id, module),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT result_json FROM analysis_results WHERE pitch_id = ?",
                    (pitch_id,),
                ).fetchall()
        return [json.loads(r["result_json"]) for r in rows]
