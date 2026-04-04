"""Pipeline package for video extraction and SAM 3D Body inference."""

from __future__ import annotations

from backend.app.pipeline.segment import PitchSegment
from backend.app.pipeline.video import FrameExtractor, SourceType

__all__ = [
    "FrameExtractor",
    "SourceType",
    "PitchSegment",
]
