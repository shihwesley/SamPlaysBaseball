"""Tests for diagnostic report engine — provider-agnostic."""

from pathlib import Path

import numpy as np
import pytest

from backend.app.reports.diagnostic import (
    DiagnosticEngine,
    DiagnosticProvider,
    DiagnosticReport,
    create_provider,
    _build_prompt,
    _compute_confidence,
    _parse_response,
)
from backend.app.reports.norms import NORMS, flag_concerns
from backend.app.analysis.compare_deliveries import compare_deliveries


# ---------------------------------------------------------------------------
# Mock provider for testing
# ---------------------------------------------------------------------------

class MockProvider(DiagnosticProvider):
    name = "mock"
    supports_vision = True

    def __init__(self, response: str = ""):
        self._response = response

    def generate(self, prompt, images=None, max_tokens=1200):
        self._last_prompt = prompt
        self._last_images = images
        return self._response


MOCK_RESPONSE = """\
Ohtani's trunk rotation dropped 13 degrees between the first and fifth innings. \
His kinetic chain sequencing broke down — the wrist peaked before the elbow, \
leaking energy. Shoulder external rotation fell to 148 degrees, still within \
normal range but trending toward the concern threshold of 140.

The velocity drop from 97.2 to 95.1 mph maps directly to the reduced trunk \
rotation. Less rotational energy means the arm has to compensate, which explains \
the command loss — zone percentage dropped from 48% to 31%.

Hip-shoulder separation at foot plant measured 29 degrees, below the 35-degree \
minimum. This suggests early trunk rotation, rushing the delivery. The stride \
length also shortened from 0.84 to 0.79 of body height.

The acceleration phase diverged most — 78% through the delivery, consistent \
with fatigue-driven mechanical breakdown.

1. Monitor trunk rotation inning-by-inning as a fatigue indicator
2. Address hip mobility to restore hip-shoulder separation above 35 degrees
3. Compare next start's first-inning mechanics to establish a fresh baseline
4. Review between-start recovery protocol — shortened stride suggests leg fatigue
5. If trunk rotation stays below 20 degrees by the 4th inning, consider earlier hook
"""


def _synthetic_joints(n=60, seed=0):
    rng = np.random.RandomState(seed)
    base = rng.randn(70, 3).astype(np.float64) * 0.3
    joints = np.tile(base, (n, 1, 1))
    t = np.linspace(0, np.pi, n)
    for j in [6, 8, 41]:
        joints[:, j, 0] += np.sin(t) * 0.3
        joints[:, j, 1] -= np.sin(t * 2) * 0.1
    joints[:, 13, 2] -= np.linspace(0, 0.5, n)
    joints[:, 13, 1] -= np.abs(np.sin(t)) * 0.15
    return joints


class TestDiagnosticEngine:
    def test_generate_returns_report(self):
        provider = MockProvider(MOCK_RESPONSE)
        engine = DiagnosticEngine(provider)
        comparison = compare_deliveries(
            _synthetic_joints(60, 0), _synthetic_joints(60, 1), fps=30.0,
        )
        report = engine.generate(
            comparison,
            concern="velocity drop",
            pitcher_name="Ohtani",
            pitches_analyzed=25,
        )
        assert isinstance(report, DiagnosticReport)
        assert report.provider == "mock"
        assert report.confidence == "high"
        assert len(report.recommendations) > 0
        assert "trunk rotation" in report.narrative.lower()

    def test_low_confidence_few_pitches(self):
        provider = MockProvider(MOCK_RESPONSE)
        engine = DiagnosticEngine(provider)
        comparison = compare_deliveries(
            _synthetic_joints(60, 0), _synthetic_joints(60, 1), fps=30.0,
        )
        report = engine.generate(comparison, pitches_analyzed=3)
        assert report.confidence == "low"

    def test_moderate_confidence(self):
        provider = MockProvider(MOCK_RESPONSE)
        engine = DiagnosticEngine(provider)
        comparison = compare_deliveries(
            _synthetic_joints(60, 0), _synthetic_joints(60, 1), fps=30.0,
        )
        report = engine.generate(comparison, pitches_analyzed=10)
        assert report.confidence == "moderate"

    def test_images_passed_to_vision_provider(self):
        provider = MockProvider(MOCK_RESPONSE)
        engine = DiagnosticEngine(provider)
        comparison = compare_deliveries(
            _synthetic_joints(60, 0), _synthetic_joints(60, 1), fps=30.0,
        )
        fake_images = ["/tmp/frame1.png", "/tmp/frame2.png"]
        report = engine.generate(comparison, images=fake_images)
        assert provider._last_images == fake_images
        assert report.images_used == 2

    def test_images_not_passed_to_text_only_provider(self):
        provider = MockProvider(MOCK_RESPONSE)
        provider.supports_vision = False
        engine = DiagnosticEngine(provider)
        comparison = compare_deliveries(
            _synthetic_joints(60, 0), _synthetic_joints(60, 1), fps=30.0,
        )
        report = engine.generate(comparison, images=["/tmp/img.png"])
        assert provider._last_images is None

    def test_concern_appears_in_prompt(self):
        provider = MockProvider(MOCK_RESPONSE)
        engine = DiagnosticEngine(provider)
        comparison = compare_deliveries(
            _synthetic_joints(60, 0), _synthetic_joints(60, 1), fps=30.0,
        )
        engine.generate(comparison, concern="slider getting crushed")
        assert "slider getting crushed" in provider._last_prompt

    def test_statcast_appears_in_prompt(self):
        provider = MockProvider(MOCK_RESPONSE)
        engine = DiagnosticEngine(provider)
        comparison = compare_deliveries(
            _synthetic_joints(60, 0), _synthetic_joints(60, 1), fps=30.0,
        )
        statcast = {"velo_a": 97.2, "velo_b": 95.1}
        engine.generate(comparison, statcast=statcast)
        assert "97.2" in provider._last_prompt


class TestParseResponse:
    def test_splits_narrative_and_recs(self):
        narrative, recs = _parse_response(MOCK_RESPONSE)
        assert len(recs) == 5
        assert "Monitor" in recs[0]
        assert "trunk rotation" in narrative.lower()

    def test_no_recs(self):
        narrative, recs = _parse_response("Just a paragraph with no numbered items.")
        assert narrative == "Just a paragraph with no numbered items."
        assert recs == []

    def test_caps_at_5_recs(self):
        text = "Narrative.\n" + "\n".join(f"{i}. Rec {i}" for i in range(1, 10))
        _, recs = _parse_response(text)
        assert len(recs) == 5


class TestNorms:
    def test_flag_low_shoulder_er(self):
        changes = [{"name": "Max shoulder ER", "late": 130, "unit": "deg"}]
        flags = flag_concerns(changes)
        assert any("below" in f.lower() for f in flags)

    def test_no_flag_healthy_value(self):
        changes = [{"name": "Max shoulder ER", "late": 165, "unit": "deg"}]
        flags = flag_concerns(changes)
        assert len(flags) == 0

    def test_flag_low_hip_shoulder_sep(self):
        changes = [{"name": "Hip-shoulder sep", "late": 20, "unit": "deg"}]
        flags = flag_concerns(changes)
        assert any("hip-shoulder" in f.lower() for f in flags)


class TestConfidence:
    def test_high(self):
        assert _compute_confidence(20) == "high"
        assert _compute_confidence(100) == "high"

    def test_moderate(self):
        assert _compute_confidence(5) == "moderate"
        assert _compute_confidence(19) == "moderate"

    def test_low(self):
        assert _compute_confidence(4) == "low"
        assert _compute_confidence(1) == "low"


class TestProviderFactory:
    def test_create_gemma4(self):
        # Just test construction, not loading
        p = create_provider("gemma4")
        assert p.name == "gemma4"
        assert p.supports_vision is True

    def test_create_claude(self):
        p = create_provider("claude")
        assert p.name == "claude"

    def test_create_ollama(self):
        p = create_provider("ollama", model="llama3")
        assert p.name == "openai-compat"

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            create_provider("nonexistent")
