"""Video frame extraction for multiple source types."""

from __future__ import annotations

import json
import subprocess
from enum import StrEnum
from pathlib import Path

import numpy as np


class SourceType(StrEnum):
    bullpen = "bullpen"
    broadcast = "broadcast"
    smartphone = "smartphone"
    milb = "milb"


_SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi"}


class FrameExtractor:
    """Extract frames from video files at a target FPS via FFmpeg."""

    def __init__(self, source_type: SourceType = SourceType.bullpen) -> None:
        self.source_type = source_type

    def _probe(self, video_path: Path) -> tuple[int, int]:
        """Return (width, height) via ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        stream = info["streams"][0]
        return int(stream["width"]), int(stream["height"])

    def extract_frames(
        self,
        video_path: str | Path,
        target_fps: int = 30,
    ) -> list[dict]:
        """Extract frames from video at target_fps.

        Returns list of dicts with keys:
          frame_index (int), timestamp_s (float), frame (np.ndarray H,W,3 RGB)
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")
        if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported extension '{path.suffix}'. "
                f"Must be one of: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
            )

        width, height = self._probe(path)

        cmd = [
            "ffmpeg",
            "-i", str(path),
            "-vf", f"fps={target_fps}",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "pipe:1",
        ]
        result = subprocess.run(cmd, capture_output=True, check=True)
        raw = result.stdout

        frame_size = width * height * 3
        if frame_size == 0 or len(raw) % frame_size != 0:
            return []

        n_frames = len(raw) // frame_size
        frames: list[dict] = []
        for i in range(n_frames):
            chunk = raw[i * frame_size : (i + 1) * frame_size]
            frame = np.frombuffer(chunk, dtype=np.uint8).reshape(height, width, 3).copy()
            frames.append({
                "frame_index": i,
                "timestamp_s": i / target_fps,
                "frame": frame,
            })

        return frames
