"""Query orchestrator — chains parser → fetch → compare → diagnostic → GLB.

Single entry point: QueryOrchestrator.execute(text) returns a response
bundle or a progress token if MLX inference is needed.

Usage:
    orch = QueryOrchestrator(db=pitch_db, parser=parser, diagnostic_engine=engine)
    result = orch.execute("Compare Ohtani's 1st inning FF to 6th inning FF")
"""

from __future__ import annotations

import logging
import statistics
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from backend.app.analysis.compare_deliveries import compare_deliveries, DeliveryComparison
from backend.app.data.pitch_db import PitchDB, MeshData
from backend.app.query.parser import AnalysisQuery, QueryParser
from backend.app.reports.diagnostic import DiagnosticEngine, DiagnosticReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

@dataclass
class StatcastGroup:
    """Aggregated Statcast stats for one pitch group."""

    avg_velo: float | None = None
    avg_spin: float | None = None
    whiff_pct: float | None = None
    zone_pct: float | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class QueryResult:
    """Full response bundle from the orchestrator."""

    report: DiagnosticReport
    comparison: DeliveryComparison
    statcast: dict
    viewer: dict
    query: dict
    pitches_used: list[str] = field(default_factory=list)


@dataclass
class ProgressToken:
    """Returned when MLX inference is needed for uncached pitches."""

    token: str
    status: str = "processing"
    pitches_needing_inference: list[str] = field(default_factory=list)
    total_pitches: int = 0


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class QueryOrchestrator:
    """Chains query parsing through to final response bundle."""

    def __init__(
        self,
        db: PitchDB,
        parser: QueryParser,
        diagnostic_engine: DiagnosticEngine,
        glb_output_dir: str | Path = "data/exports",
    ) -> None:
        self.db = db
        self.parser = parser
        self.engine = diagnostic_engine
        self.glb_dir = Path(glb_output_dir)
        self.glb_dir.mkdir(parents=True, exist_ok=True)

    def execute(
        self,
        text: str,
        parsed: AnalysisQuery | None = None,
    ) -> QueryResult | ProgressToken:
        """Execute a full query pipeline.

        Args:
            text: Raw analyst query text.
            parsed: Pre-parsed query (skips parser step). Useful for retries.

        Returns:
            QueryResult if all meshes cached, ProgressToken if inference needed.
        """
        # 1. Parse
        if parsed is None:
            grounding = self._build_grounding()
            parsed = self.parser.parse(text, available_pitchers=grounding)

        errors = parsed.validate()
        if errors:
            raise ValueError(f"Invalid query: {'; '.join(errors)}")

        # 2. Resolve pitcher
        pitcher_id, pitcher_name, handedness = self._resolve_pitcher(parsed.pitcher_name)

        # 3. Fetch and split pitches
        group_a, group_b = self._fetch_pitch_groups(parsed, pitcher_id)

        total_pitches = len(group_a) + len(group_b)
        if len(group_a) < 1 or len(group_b) < 1:
            raise ValueError(
                f"Not enough pitches: group A has {len(group_a)}, group B has {len(group_b)}"
            )

        # 4. Check mesh availability
        all_pitches = group_a + group_b
        uncached = [p for p in all_pitches if not p.get("mesh_path")]
        if uncached:
            token = str(uuid.uuid4())[:12]
            return ProgressToken(
                token=f"q_{token}",
                pitches_needing_inference=[p["play_id"] for p in uncached],
                total_pitches=total_pitches,
            )

        # 5. Select representative pitches
        rep_a = self._select_representative(group_a, parsed.comparison_mode, "a")
        rep_b = self._select_representative(group_b, parsed.comparison_mode, "b")

        # 6. Load meshes and compare
        mesh_a = self.db.load_mesh(rep_a["play_id"])
        mesh_b = self.db.load_mesh(rep_b["play_id"])

        labels = self._make_labels(parsed, rep_a, rep_b)

        comparison = compare_deliveries(
            joints_a=mesh_a.joints_mhr70,
            joints_b=mesh_b.joints_mhr70,
            fps=60.0,
            handedness=handedness,
            labels=labels,
        )

        # 7. Statcast aggregation
        statcast = {
            "group_a": _aggregate_statcast(group_a).to_dict(),
            "group_b": _aggregate_statcast(group_b).to_dict(),
        }

        # 8. Diagnostic report
        pitches_analyzed = total_pitches
        report = self.engine.generate(
            comparison=comparison,
            statcast=statcast,
            concern=parsed.concern,
            pitcher_name=pitcher_name,
            handedness=handedness,
            pitches_analyzed=pitches_analyzed,
        )

        # 9. GLB export
        comparison_id = str(uuid.uuid4())[:8]
        glb_path = self.glb_dir / f"{comparison_id}.glb"
        comparison.export_glb(
            str(glb_path),
            mesh_frames_a=_load_vertex_frames(self.db, rep_a["play_id"]),
            mesh_frames_b=_load_vertex_frames(self.db, rep_b["play_id"]),
        )

        # Phase markers from the representative pitch A
        phase_markers = {}
        if comparison.features_a and comparison.features_a.phases:
            ph = comparison.features_a.phases
            phase_markers = {
                "foot_plant": ph.foot_plant,
                "mer": ph.mer,
                "release": ph.release,
            }

        viewer = {
            "glb_url": f"/api/export/glb/{comparison_id}",
            "phase_markers": phase_markers,
            "total_frames": mesh_a.joints_mhr70.shape[0],
        }

        return QueryResult(
            report=report,
            comparison=comparison,
            statcast=statcast,
            viewer=viewer,
            query={
                "raw_text": text,
                "parsed": {
                    "pitcher_name": parsed.pitcher_name,
                    "pitch_types": parsed.pitch_types,
                    "comparison_mode": parsed.comparison_mode,
                    "concern": parsed.concern,
                },
                "pitches_used": [rep_a["play_id"], rep_b["play_id"]],
            },
            pitches_used=[rep_a["play_id"], rep_b["play_id"]],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_grounding(self) -> list[dict]:
        """Build grounding context from PitchDB for the parser."""
        rows = self.db.query(
            "SELECT DISTINCT pitcher_name, pitcher_id FROM pitches WHERE pitcher_name IS NOT NULL"
        )
        grounding = []
        for r in rows:
            types = self.db.query(
                "SELECT DISTINCT pitch_type FROM pitches WHERE pitcher_id = ? AND pitch_type IS NOT NULL",
                (r["pitcher_id"],),
            )
            games = self.db.query(
                "SELECT DISTINCT game_date FROM pitches WHERE pitcher_id = ? ORDER BY game_date DESC LIMIT 5",
                (r["pitcher_id"],),
            )
            grounding.append({
                "name": r["pitcher_name"],
                "pitch_types": [t["pitch_type"] for t in types],
                "games": [g["game_date"] for g in games],
            })
        return grounding

    def _resolve_pitcher(self, name: str) -> tuple[int, str, str]:
        """Resolve pitcher name to (pitcher_id, full_name, handedness).

        Tries exact match first, then case-insensitive LIKE.
        """
        # Exact match
        rows = self.db.query(
            "SELECT DISTINCT pitcher_id, pitcher_name FROM pitches WHERE pitcher_name = ?",
            (name,),
        )
        if not rows:
            # Case-insensitive partial match
            rows = self.db.query(
                "SELECT DISTINCT pitcher_id, pitcher_name FROM pitches WHERE pitcher_name LIKE ?",
                (f"%{name}%",),
            )
        if not rows:
            raise ValueError(f"Pitcher not found: {name!r}")
        if len(rows) > 1:
            names = [r["pitcher_name"] for r in rows]
            raise ValueError(f"Ambiguous pitcher name {name!r}: matches {names}")

        pid = rows[0]["pitcher_id"]
        full_name = rows[0]["pitcher_name"]
        # Default to right-handed; the DB could store this but doesn't currently
        handedness = "right"
        return pid, full_name, handedness

    def _fetch_pitch_groups(
        self, query: AnalysisQuery, pitcher_id: int,
    ) -> tuple[list[dict], list[dict]]:
        """Fetch and split pitches into two groups per comparison mode."""
        # Base filter: pitcher + game
        base_where = "pitcher_id = ?"
        base_params: list = [pitcher_id]

        if query.game_date:
            base_where += " AND game_date = ?"
            base_params.append(query.game_date)
        else:
            # Most recent game
            row = self.db.query(
                f"SELECT DISTINCT game_date FROM pitches WHERE {base_where} ORDER BY game_date DESC LIMIT 1",
                tuple(base_params),
            )
            if row:
                base_where += " AND game_date = ?"
                base_params.append(row[0]["game_date"])

        if query.comparison_mode == "time":
            # Same pitch type, split by innings
            pt = query.pitch_types[0] if query.pitch_types else None
            type_filter = f" AND pitch_type = ?" if pt else ""
            type_params = [pt] if pt else []

            all_pitches = self.db.query(
                f"SELECT * FROM pitches WHERE {base_where}{type_filter} ORDER BY inning",
                tuple(base_params + type_params),
            )

            ra, rb = query.inning_range_a, query.inning_range_b
            group_a = [p for p in all_pitches if ra and ra[0] <= p["inning"] <= ra[1]]
            group_b = [p for p in all_pitches if rb and rb[0] <= p["inning"] <= rb[1]]

        elif query.comparison_mode == "type":
            # Different pitch types
            type_a, type_b = query.pitch_types[0], query.pitch_types[1]
            group_a = self.db.query(
                f"SELECT * FROM pitches WHERE {base_where} AND pitch_type = ?",
                tuple(base_params + [type_a]),
            )
            group_b = self.db.query(
                f"SELECT * FROM pitches WHERE {base_where} AND pitch_type = ?",
                tuple(base_params + [type_b]),
            )

        elif query.comparison_mode == "baseline":
            # Current vs baseline (all pitches of that type)
            pt = query.pitch_types[0] if query.pitch_types else None
            type_filter = f" AND pitch_type = ?" if pt else ""
            type_params = [pt] if pt else []

            all_pitches = self.db.query(
                f"SELECT * FROM pitches WHERE pitcher_id = ?{type_filter} ORDER BY game_date DESC",
                tuple([pitcher_id] + type_params),
            )

            if not all_pitches:
                return [], []

            # Group A: the most recent game's pitches (already filtered above)
            recent_date = all_pitches[0]["game_date"]
            group_a = [p for p in all_pitches if p["game_date"] == recent_date]
            # Group B: everything else (the baseline pool)
            group_b = [p for p in all_pitches if p["game_date"] != recent_date]

        else:
            raise ValueError(f"Unknown comparison_mode: {query.comparison_mode!r}")

        return group_a, group_b

    def _select_representative(
        self, pitches: list[dict], mode: str, group: str,
    ) -> dict:
        """Pick one representative pitch from a group.

        - time: median pitch by inning
        - type: first pitch with mesh data
        - baseline: random from pool (or first with mesh)
        """
        with_mesh = [p for p in pitches if p.get("mesh_path")]
        pool = with_mesh if with_mesh else pitches

        if mode == "time":
            # Pick median inning pitch
            pool_sorted = sorted(pool, key=lambda p: p["inning"])
            return pool_sorted[len(pool_sorted) // 2]

        # type and baseline: first available
        return pool[0]

    def _make_labels(
        self, query: AnalysisQuery, rep_a: dict, rep_b: dict,
    ) -> tuple[str, str]:
        """Create display labels for the two comparison groups."""
        if query.comparison_mode == "time":
            pt = query.pitch_types[0] if query.pitch_types else "??"
            ra = query.inning_range_a
            rb = query.inning_range_b
            label_a = f"{pt} Inn {ra[0]}-{ra[1]}" if ra else f"{pt} early"
            label_b = f"{pt} Inn {rb[0]}-{rb[1]}" if rb else f"{pt} late"
        elif query.comparison_mode == "type":
            label_a = f"{rep_a.get('pitch_type', '??')} Inn {rep_a.get('inning', '?')}"
            label_b = f"{rep_b.get('pitch_type', '??')} Inn {rep_b.get('inning', '?')}"
        else:
            label_a = f"Latest ({rep_a.get('game_date', '?')})"
            label_b = "Baseline"
        return label_a, label_b


# ---------------------------------------------------------------------------
# Statcast aggregation
# ---------------------------------------------------------------------------

def _aggregate_statcast(pitches: list[dict]) -> StatcastGroup:
    """Compute group-level Statcast averages."""
    velos = [p["release_speed"] for p in pitches if p.get("release_speed")]
    spins = [p["release_spin_rate"] for p in pitches if p.get("release_spin_rate")]

    # Whiff rate: swinging_strike / total swings approximation
    total = len(pitches)
    swinging_strikes = sum(
        1 for p in pitches
        if p.get("description") and "swinging" in str(p["description"]).lower()
        and "strike" in str(p["description"]).lower()
    )
    called_or_in_zone = sum(
        1 for p in pitches
        if p.get("plate_x") is not None and p.get("plate_z") is not None
        and abs(p["plate_x"]) <= 0.83 and 1.5 <= p["plate_z"] <= 3.5
    )

    return StatcastGroup(
        avg_velo=statistics.mean(velos) if velos else None,
        avg_spin=statistics.mean(spins) if spins else None,
        whiff_pct=round(swinging_strikes / total * 100, 1) if total > 0 else None,
        zone_pct=round(called_or_in_zone / total * 100, 1) if total > 0 else None,
    )


def _load_vertex_frames(db: PitchDB, play_id: str) -> list[np.ndarray] | None:
    """Load mesh vertex frames for GLB export."""
    mesh = db.load_mesh(play_id)
    if mesh is None:
        return None
    # Split (T, N, 3) into list of (N, 3)
    return [mesh.vertices[i] for i in range(mesh.vertices.shape[0])]
