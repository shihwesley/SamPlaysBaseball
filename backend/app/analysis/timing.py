"""Timing analysis: kinetic chain sequence validation and energy decomposition."""

from __future__ import annotations

import numpy as np

from backend.app.models.analysis import TimingAnalysisResult, TimingEvent
from backend.app.pipeline.features import BiomechFeatures

# Normative gaps: ~9.5ms per segment at 30fps ~ 0.285 frames; use 2 frames as min gap
_MIN_GAP_FRAMES = 2


class TimingAnalyzer:
    """Validate kinetic chain timing sequence and detect issues."""

    def analyze(
        self,
        pitch_id: str,
        pitcher_id: str,
        features: BiomechFeatures,
        pitch_type: str,
        fps: float = 30.0,
    ) -> TimingAnalysisResult:
        """Return timing analysis result."""
        kc = features.kinetic_chain
        phases = features.phases

        def ms(frame: int) -> float:
            return frame / fps * 1000.0

        events: list[TimingEvent] = [
            TimingEvent(
                event_name="foot_plant",
                frame=phases.foot_plant,
                time_ms=ms(phases.foot_plant),
            ),
            TimingEvent(
                event_name="pelvis_peak",
                frame=kc.pelvis_peak_frame,
                time_ms=ms(kc.pelvis_peak_frame),
            ),
            TimingEvent(
                event_name="trunk_peak",
                frame=kc.trunk_peak_frame,
                time_ms=ms(kc.trunk_peak_frame),
            ),
            TimingEvent(
                event_name="shoulder_peak",
                frame=kc.shoulder_peak_frame,
                time_ms=ms(kc.shoulder_peak_frame),
            ),
            TimingEvent(
                event_name="elbow_peak",
                frame=kc.elbow_peak_frame,
                time_ms=ms(kc.elbow_peak_frame),
            ),
            TimingEvent(
                event_name="wrist_peak",
                frame=kc.wrist_peak_frame,
                time_ms=ms(kc.wrist_peak_frame),
            ),
            TimingEvent(
                event_name="mer",
                frame=phases.mer,
                time_ms=ms(phases.mer),
            ),
            TimingEvent(
                event_name="ball_release",
                frame=phases.release,
                time_ms=ms(phases.release),
            ),
        ]

        # Check sequence validity
        sequence_valid = kc.sequence_valid

        # Check gaps between adjacent segments
        segment_frames = [
            kc.pelvis_peak_frame,
            kc.trunk_peak_frame,
            kc.shoulder_peak_frame,
            kc.elbow_peak_frame,
            kc.wrist_peak_frame,
        ]
        gaps = [
            segment_frames[i + 1] - segment_frames[i]
            for i in range(len(segment_frames) - 1)
        ]
        any_gap_too_small = any(g < _MIN_GAP_FRAMES for g in gaps)

        is_timing_issue = not sequence_valid or any_gap_too_small

        # Timing score: start at 1.0, penalize each inversion
        inversions = sum(
            1 for i in range(len(segment_frames) - 1)
            if segment_frames[i] > segment_frames[i + 1]
        )
        timing_score = max(0.0, 1.0 - 0.25 * inversions)

        return TimingAnalysisResult(
            pitch_id=pitch_id,
            pitcher_id=pitcher_id,
            pitch_type=pitch_type,
            events=events,
            timing_score=timing_score,
            is_timing_issue=is_timing_issue,
        )
