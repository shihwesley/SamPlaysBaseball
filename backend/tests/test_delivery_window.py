"""Tests for the delivery-window trim detector.

Slow tests. These load SAM 3.1 (~5s cold) and run end-to-end detection
on two real Darvish clips that live under data/clips/526517/. Marked as
an integration test so the fast unit run can skip them.

Run just these:
    pytest backend/tests/test_delivery_window.py -v

Skip them in the fast suite:
    pytest -m "not integration"
"""

from pathlib import Path

import pytest

from backend.app.pipeline.delivery_window import (
    DeliveryWindow,
    find_delivery_window,
    write_trimmed_clip,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CLIPS_DIR = REPO_ROOT / "data" / "clips" / "526517"

IN_PLAY_CLIP = CLIPS_DIR / "inn1_ab4_p1_CH_b69a4fbd.mp4"  # broadcast cut at 155
BALL_CLIP = CLIPS_DIR / "inn1_ab5_p1_CH_dd857f31.mp4"    # no cut, biomech cap


pytestmark = pytest.mark.integration


def _require(p: Path):
    if not p.exists():
        pytest.skip(f"fixture missing: {p}")


class TestFindDeliveryWindow:
    def test_in_play_clip_trims_at_broadcast_cut(self):
        """Darvish CH in-play: scene cut at frame 155 should drive end_frame."""
        _require(IN_PLAY_CLIP)
        window = find_delivery_window(IN_PLAY_CLIP)

        assert isinstance(window, DeliveryWindow)
        assert window.set_frame is not None
        assert window.set_frame < 30  # pitcher already in set in the first second
        assert window.scene_cut_frame == 155
        assert window.end_frame == 155
        assert window.confidence >= 0.5
        # Early-exit should keep the SAM call count tiny
        assert len(window.sampled) <= 6, (
            f"SAM early-exit regression: {len(window.sampled)} calls"
        )

    def test_ball_clip_trims_at_biomech_cap(self):
        """Darvish CH ball: no broadcast cut, biomechanical 5s cap rules."""
        _require(BALL_CLIP)
        window = find_delivery_window(BALL_CLIP)

        assert window.set_frame is not None
        assert window.scene_cut_frame is None
        # 5s @ 29.97 fps = ~150 frames
        assert window.end_frame == 150
        assert window.start_frame == 0
        assert window.n_frames == 150
        assert window.confidence >= 0.5
        assert len(window.sampled) <= 4

    def test_to_dict_roundtrip(self):
        """Serialized form must include everything the fetch script reads."""
        _require(BALL_CLIP)
        window = find_delivery_window(BALL_CLIP)
        d = window.to_dict()
        for key in (
            "start_frame",
            "end_frame",
            "total_frames",
            "fps",
            "set_frame",
            "scene_cut_frame",
            "n_frames_kept",
            "duration_kept_s",
            "confidence",
            "n_sam_calls",
        ):
            assert key in d, f"missing key in to_dict: {key}"


class TestWriteTrimmedClip:
    def test_writes_exact_frame_count(self, tmp_path):
        _require(BALL_CLIP)
        out = tmp_path / "trimmed.mp4"
        n = write_trimmed_clip(BALL_CLIP, out, start_frame=0, end_frame=90)
        assert n == 90
        assert out.exists()
        assert out.stat().st_size > 10_000  # real video, not an HTML error page

    def test_empty_range_raises(self, tmp_path):
        _require(BALL_CLIP)
        out = tmp_path / "trimmed.mp4"
        with pytest.raises(RuntimeError):
            write_trimmed_clip(BALL_CLIP, out, start_frame=0, end_frame=0)
