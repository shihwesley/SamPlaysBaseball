"""AnalysisResult models for the 6 analysis modules."""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class AnalysisResult(BaseModel):
    """Base class for all analysis results."""

    pitch_id: str
    pitcher_id: str
    module: str
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    notes: str | None = None


# ---------------------------------------------------------------------------
# 1. Baseline Comparison
# ---------------------------------------------------------------------------


class JointDeviation(BaseModel):
    joint_index: int
    joint_name: str | None = None
    deviation_mm: float
    z_score: float


class BaselineComparisonResult(AnalysisResult):
    module: Literal["baseline-comparison"] = "baseline-comparison"
    pitch_type: str
    # Overall deviation score: mean z-score across all joints
    overall_z_score: float
    # Top joints that deviate most from baseline
    top_deviations: list[JointDeviation] = Field(default_factory=list)
    # Whether this pitch is a statistical outlier
    is_outlier: bool = False
    outlier_threshold: float = 2.5


# ---------------------------------------------------------------------------
# 2. Tipping Detection
# ---------------------------------------------------------------------------


class TipSignal(BaseModel):
    """A detected mechanical signal that may tip a pitch type."""

    feature_name: str
    pitch_type_a: str
    pitch_type_b: str
    # How distinguishable are these two pitch types on this feature
    separation_score: float  # 0 = indistinguishable, 1 = fully separable
    frame_range: tuple[int, int] | None = None


class TippingDetectionResult(AnalysisResult):
    module: Literal["tipping-detection"] = "tipping-detection"
    # Detected tip signals with separation scores
    tip_signals: list[TipSignal] = Field(default_factory=list)
    # Highest risk signal
    max_separation_score: float = 0.0
    is_tipping: bool = False
    tipping_threshold: float = 0.7


# ---------------------------------------------------------------------------
# 3. Fatigue Tracking
# ---------------------------------------------------------------------------


class FatigueMarker(BaseModel):
    metric_name: str
    value: float
    baseline_value: float
    pct_change: float


class FatigueTrackingResult(AnalysisResult):
    module: Literal["fatigue-tracking"] = "fatigue-tracking"
    pitch_number_in_game: int
    # Fatigue markers (arm slot drop, velocity proxy, etc.)
    markers: list[FatigueMarker] = Field(default_factory=list)
    # Composite fatigue score 0-1
    fatigue_score: float = Field(ge=0.0, le=1.0, default=0.0)
    is_fatigued: bool = False
    fatigue_threshold: float = 0.6


# ---------------------------------------------------------------------------
# 4. Command Analysis
# ---------------------------------------------------------------------------


class CommandAnalysisResult(AnalysisResult):
    module: Literal["command-analysis"] = "command-analysis"
    pitch_type: str
    # Plate location
    plate_x: float | None = None
    plate_z: float | None = None
    # Release point consistency vs baseline
    release_x_deviation: float | None = None  # mm
    release_z_deviation: float | None = None  # mm
    # Command score: 0 = worst, 1 = best
    command_score: float = Field(ge=0.0, le=1.0, default=0.5)
    # Zone classification
    zone: int | None = None  # Statcast zone 1-9, 11-14


# ---------------------------------------------------------------------------
# 5. Arm Slot Drift
# ---------------------------------------------------------------------------


class ArmSlotDriftResult(AnalysisResult):
    module: Literal["arm-slot-drift"] = "arm-slot-drift"
    # Arm slot angle in degrees (shoulder-to-wrist vector angle)
    arm_slot_degrees: float
    baseline_arm_slot_degrees: float | None = None
    drift_degrees: float | None = None
    # Cumulative drift over game
    cumulative_drift_degrees: float | None = None
    is_significant_drift: bool = False
    drift_threshold_degrees: float = 3.0


# ---------------------------------------------------------------------------
# 6. Timing Analysis
# ---------------------------------------------------------------------------


class TimingEvent(BaseModel):
    event_name: str  # e.g. "foot_plant", "max_hip_rotation", "ball_release"
    frame: int
    time_ms: float | None = None
    baseline_frame: int | None = None
    frame_delta: int | None = None  # positive = late, negative = early


class TimingAnalysisResult(AnalysisResult):
    module: Literal["timing-analysis"] = "timing-analysis"
    pitch_type: str
    events: list[TimingEvent] = Field(default_factory=list)
    # Overall timing consistency score 0-1
    timing_score: float = Field(ge=0.0, le=1.0, default=1.0)
    is_timing_issue: bool = False


# ---------------------------------------------------------------------------
# Union type for polymorphic deserialization
# ---------------------------------------------------------------------------

AnyAnalysisResult = Annotated[
    Union[
        BaselineComparisonResult,
        TippingDetectionResult,
        FatigueTrackingResult,
        CommandAnalysisResult,
        ArmSlotDriftResult,
        TimingAnalysisResult,
    ],
    Field(discriminator="module"),
]
