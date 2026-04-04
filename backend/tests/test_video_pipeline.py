"""Tests for the video pipeline (no real video files or GPU required)."""

from __future__ import annotations

import numpy as np
import pytest

from backend.app.pipeline.isolate import isolate_pitcher
from backend.app.pipeline.preprocess import preprocess_frame
from backend.app.pipeline.segment import PitchSegment, segment_pitch
from backend.app.pipeline.video import FrameExtractor, SourceType


# ---------------------------------------------------------------------------
# SourceType
# ---------------------------------------------------------------------------

def test_source_type_has_all_variants():
    assert SourceType.bullpen == "bullpen"
    assert SourceType.broadcast == "broadcast"
    assert SourceType.smartphone == "smartphone"
    assert SourceType.milb == "milb"


# ---------------------------------------------------------------------------
# PitchSegment
# ---------------------------------------------------------------------------

def test_pitch_segment_is_dataclass():
    seg = PitchSegment(frame_start=5, frame_end=25, confidence=0.8)
    assert seg.frame_start == 5
    assert seg.frame_end == 25
    assert seg.confidence == 0.8


# ---------------------------------------------------------------------------
# preprocess_frame
# ---------------------------------------------------------------------------

_SAMPLE_FRAME = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)


@pytest.mark.parametrize("source_type", list(SourceType))
def test_preprocess_returns_same_shape(source_type):
    result = preprocess_frame(_SAMPLE_FRAME, source_type)
    assert result.shape == _SAMPLE_FRAME.shape


# ---------------------------------------------------------------------------
# segment_pitch
# ---------------------------------------------------------------------------

def _make_frames(pixel_values: list[int]) -> list[dict]:
    """Build minimal fake frames with the given uniform pixel values."""
    return [
        {"frame_index": i, "timestamp_s": i / 30.0, "frame": np.full((64, 64, 3), v, dtype=np.uint8)}
        for i, v in enumerate(pixel_values)
    ]


def test_segment_pitch_detects_single_pitch():
    # 30 quiet frames, 10 high-motion frames, 30 quiet frames
    quiet = [10] * 30
    motion = list(range(10, 220, 20))  # 10 frames with rising pixel values (simulate motion)
    quiet2 = [10] * 30
    frames = _make_frames(quiet + motion + quiet2)

    segments = segment_pitch(frames, fps=30)
    assert len(segments) == 1, f"Expected 1 segment, got {len(segments)}"
    seg = segments[0]
    assert seg.frame_start >= 0
    assert seg.frame_end < len(frames)
    assert 0.0 <= seg.confidence <= 1.0


def test_segment_pitch_returns_empty_for_flat_frames():
    frames = _make_frames([10] * 50)
    segments = segment_pitch(frames, fps=30)
    assert segments == []


def test_segment_pitch_sorted_by_frame_start():
    # Two motion bursts separated by quiet
    pixel_seq = [10] * 10 + [200, 10, 200, 10, 200] + [10] * 20 + [200, 10, 200, 10, 200] + [10] * 10
    frames = _make_frames(pixel_seq)
    segments = segment_pitch(frames, fps=30)
    starts = [s.frame_start for s in segments]
    assert starts == sorted(starts)


# ---------------------------------------------------------------------------
# isolate_pitcher
# ---------------------------------------------------------------------------

def test_isolate_pitcher_center_crop_width():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cropped = isolate_pitcher(frame)
    expected_w = int(640 * 0.60)
    assert cropped.shape[0] == 480
    assert cropped.shape[1] == expected_w
    assert cropped.shape[2] == 3


def test_isolate_pitcher_full_height():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cropped = isolate_pitcher(frame)
    assert cropped.shape[0] == 720


# ---------------------------------------------------------------------------
# FrameExtractor validation
# ---------------------------------------------------------------------------

def test_frame_extractor_missing_file():
    extractor = FrameExtractor()
    with pytest.raises(FileNotFoundError):
        extractor.extract_frames("/nonexistent/path/video.mp4")


def test_frame_extractor_unsupported_extension(tmp_path):
    fake = tmp_path / "clip.txt"
    fake.write_text("not a video")
    extractor = FrameExtractor()
    with pytest.raises(ValueError, match="Unsupported extension"):
        extractor.extract_frames(fake)
