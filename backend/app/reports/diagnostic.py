"""Diagnostic report generator — provider-agnostic.

Supports local multimodal models (Gemma 4 via mlx-vlm), cloud APIs
(Claude, OpenAI), or any future provider. Swap backends by changing
the provider argument.

Usage:
    from backend.app.reports.diagnostic import (
        DiagnosticEngine, DiagnosticReport, create_provider,
    )

    engine = DiagnosticEngine(provider=create_provider("gemma4"))
    report = engine.generate(comparison, statcast, concern="command issues")

    # Or with cloud fallback:
    engine = DiagnosticEngine(provider=create_provider("claude"))
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from backend.app.analysis.compare_deliveries import DeliveryComparison
from backend.app.reports.norms import NORMS, flag_concerns


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticReport:
    """The final diagnostic output."""

    narrative: str
    recommendations: list[str]
    risk_flags: list[str]
    confidence: str             # "high" | "moderate" | "low"
    pitches_analyzed: int
    provider: str               # which backend generated this
    images_used: int = 0        # how many visual inputs were sent


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------

class DiagnosticProvider(ABC):
    """Abstract interface for diagnostic report generation.

    Implement this to add a new LLM backend. The engine calls generate()
    with a text prompt and optional images. The provider returns raw text
    that the engine parses into DiagnosticReport.
    """

    name: str = "base"
    supports_vision: bool = False

    @abstractmethod
    def generate(
        self,
        prompt: str,
        images: list[Path | str] | None = None,
        max_tokens: int = 1200,
    ) -> str:
        """Generate diagnostic text from prompt + optional images.

        Args:
            prompt: Full text prompt with system instructions + context
            images: Optional list of image paths (only if supports_vision)
            max_tokens: Maximum response length

        Returns:
            Raw text response from the model.
        """
        ...


# ---------------------------------------------------------------------------
# Gemma 4 E4B provider (local, multimodal, via mlx-vlm)
# ---------------------------------------------------------------------------

class Gemma4Provider(DiagnosticProvider):
    """Local Gemma 4 E4B via mlx-vlm. Vision-capable."""

    name = "gemma4"
    supports_vision = True

    def __init__(
        self,
        model_id: str = "google/gemma-4-e4b-it",
        enable_thinking: bool = True,
        max_tokens: int = 1200,
    ):
        self.model_id = model_id
        self.enable_thinking = enable_thinking
        self._default_max_tokens = max_tokens
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        from mlx_vlm import load
        self._model, self._processor = load(self.model_id)

    def generate(
        self,
        prompt: str,
        images: list[Path | str] | None = None,
        max_tokens: int = 0,
    ) -> str:
        self._load()
        from mlx_vlm import generate as vlm_generate
        from mlx_vlm.prompt_utils import apply_chat_template

        max_tok = max_tokens or self._default_max_tokens
        image_list = [str(p) for p in images] if images else None
        num_images = len(image_list) if image_list else 0

        formatted = apply_chat_template(
            self._processor,
            self._model.config,
            prompt,
            num_images=num_images,
            chat_template_kwargs={"enable_thinking": self.enable_thinking},
        )

        return vlm_generate(
            model=self._model,
            processor=self._processor,
            prompt=formatted,
            image=image_list,
            max_tokens=max_tok,
        )


# ---------------------------------------------------------------------------
# Claude API provider (cloud, text-only or vision)
# ---------------------------------------------------------------------------

class ClaudeProvider(DiagnosticProvider):
    """Anthropic Claude API. Supports vision via base64 images."""

    name = "claude"
    supports_vision = True

    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250514",
        api_key: str | None = None,
    ):
        self.model = model
        self.api_key = api_key

    def generate(
        self,
        prompt: str,
        images: list[Path | str] | None = None,
        max_tokens: int = 1200,
    ) -> str:
        import base64
        from anthropic import Anthropic

        client = Anthropic(api_key=self.api_key)

        content = []
        if images:
            for img_path in images:
                img_bytes = Path(img_path).read_bytes()
                suffix = Path(img_path).suffix.lower()
                media = {"image/png": "image/png", ".jpg": "image/jpeg",
                         ".jpeg": "image/jpeg", ".png": "image/png"}.get(suffix, "image/png")
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media,
                        "data": base64.b64encode(img_bytes).decode(),
                    },
                })
        content.append({"type": "text", "text": prompt})

        response = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        return response.content[0].text.strip()


# ---------------------------------------------------------------------------
# Generic OpenAI-compatible provider (Ollama, vLLM, OpenRouter, etc.)
# ---------------------------------------------------------------------------

class OpenAICompatProvider(DiagnosticProvider):
    """Any OpenAI-compatible API (Ollama, vLLM, OpenRouter, Together, etc.)."""

    name = "openai-compat"
    supports_vision = False  # text-only by default

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "llama3",
        api_key: str = "ollama",
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key

    def generate(
        self,
        prompt: str,
        images: list[Path | str] | None = None,
        max_tokens: int = 1200,
    ) -> str:
        from openai import OpenAI

        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PROVIDERS = {
    "gemma4": Gemma4Provider,
    "claude": ClaudeProvider,
    "openai": OpenAICompatProvider,
    "ollama": OpenAICompatProvider,
}


def create_provider(name: str, **kwargs) -> DiagnosticProvider:
    """Create a diagnostic provider by name.

    Args:
        name: "gemma4", "claude", "openai", "ollama"
        **kwargs: passed to the provider constructor

    Examples:
        create_provider("gemma4")
        create_provider("claude", model="claude-opus-4-5")
        create_provider("ollama", model="llama3", base_url="http://localhost:11434/v1")
    """
    cls = _PROVIDERS.get(name)
    if cls is None:
        raise ValueError(f"Unknown provider: {name!r}. Available: {list(_PROVIDERS)}")
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a pitching mechanics analyst with 30 years of professional \
experience across MLB player development. You have worked with dozens \
of pitchers from rookie ball to the majors. You are writing an internal \
assessment for the pitching coach and player development staff.

Rules:
- Describe what you SEE in the images first (if provided), then connect to the numbers
- Lead with what changed mechanically and why it matters for performance
- Connect mechanical changes to the Statcast outcomes the analyst sees
- Reference specific numbers — never generalize
- Use biomechanics terminology (MER, hip-shoulder separation, kinetic \
chain sequencing) but explain the implication in plain terms
- If the kinetic chain is disrupted, say so explicitly
- Flag injury risk when relevant (inverted W, low shoulder ER with \
high velocity = UCL stress, reduced hip-shoulder sep = arm-dominant)
- End with 3-5 specific, actionable items for the pitching coach
- Write 4-6 paragraphs. No headers, no bullets in the narrative. \
Recommendations as a numbered list at the end.
- Never say "based on the data" or "the analysis shows" — state \
findings directly as observations\
"""


# ---------------------------------------------------------------------------
# Diagnostic engine
# ---------------------------------------------------------------------------

class DiagnosticEngine:
    """Generates diagnostic reports from delivery comparisons.

    Provider-agnostic: pass any DiagnosticProvider.
    """

    def __init__(self, provider: DiagnosticProvider):
        self.provider = provider

    def generate(
        self,
        comparison: DeliveryComparison,
        statcast: dict | None = None,
        concern: str | None = None,
        pitcher_name: str = "Unknown",
        handedness: str = "right",
        pitches_analyzed: int = 2,
        images: list[Path | str] | None = None,
    ) -> DiagnosticReport:
        """Generate a full diagnostic report.

        Args:
            comparison: DeliveryComparison from compare_deliveries()
            statcast: Optional Statcast context dict (velo, spin, whiff%, zone%)
            concern: The analyst's original question/concern
            pitcher_name: Pitcher's name for the report
            handedness: "right" or "left"
            pitches_analyzed: Total pitches informing this comparison
            images: Optional rendered images of key frames / overlays
        """
        confidence = _compute_confidence(pitches_analyzed)

        # Build the structured context
        mechanical_changes = _build_mechanical_changes(comparison)
        risk_flags = flag_concerns(mechanical_changes)

        prompt = _build_prompt(
            concern=concern,
            pitcher=pitcher_name,
            handedness=handedness,
            comparison=comparison,
            mechanical_changes=mechanical_changes,
            statcast=statcast,
            has_images=bool(images) and self.provider.supports_vision,
        )

        # Only pass images if provider supports vision
        img_input = images if self.provider.supports_vision else None

        raw_text = self.provider.generate(prompt, images=img_input)

        narrative, recommendations = _parse_response(raw_text)

        return DiagnosticReport(
            narrative=narrative,
            recommendations=recommendations,
            risk_flags=risk_flags,
            confidence=confidence,
            pitches_analyzed=pitches_analyzed,
            provider=self.provider.name,
            images_used=len(images) if images else 0,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_confidence(n_pitches: int) -> str:
    if n_pitches >= 20:
        return "high"
    if n_pitches >= 5:
        return "moderate"
    return "low"


def _build_mechanical_changes(comparison: DeliveryComparison) -> list[dict]:
    return [
        {
            "name": d.name,
            "early": d.value_a,
            "late": d.value_b,
            "change": d.diff,
            "unit": d.unit,
        }
        for d in sorted(comparison.diffs, key=lambda x: x.abs_diff, reverse=True)
    ]


def _build_prompt(
    concern: str | None,
    pitcher: str,
    handedness: str,
    comparison: DeliveryComparison,
    mechanical_changes: list[dict],
    statcast: dict | None,
    has_images: bool,
) -> str:
    parts = [_SYSTEM_PROMPT, ""]

    if has_images:
        parts.append(
            "Images are attached showing the pitcher's 3D body reconstruction "
            "at key delivery phases. The solid body is the primary pitch, "
            "the transparent body is the comparison."
        )
        parts.append("")

    if concern:
        parts.append(f"Analyst's concern: {concern}")
        parts.append("")

    parts.append(f"Pitcher: {pitcher} ({handedness}-handed)")
    parts.append(f"Comparing: {comparison.label_a} vs {comparison.label_b}")
    parts.append("")

    # Phase timing
    pa = comparison.features_a.phases
    pb = comparison.features_b.phases
    parts.append("Phase timing:")
    parts.append(f"  Foot plant to MER: {pa.mer - pa.foot_plant} vs {pb.mer - pb.foot_plant} frames")
    parts.append(f"  MER to release: {pa.release - pa.mer} vs {pb.release - pb.mer} frames")
    parts.append("")

    # Kinetic chain
    parts.append(f"Kinetic chain ({comparison.label_a}): {_chain_str(comparison.features_a)}")
    parts.append(f"Kinetic chain ({comparison.label_b}): {_chain_str(comparison.features_b)}")
    parts.append("")

    # Mechanical changes
    parts.append("Mechanical changes (sorted by magnitude):")
    for mc in mechanical_changes[:8]:
        arrow = "+" if mc["change"] > 0 else ""
        parts.append(
            f"  {mc['name']}: {mc['early']:.1f} → {mc['late']:.1f} "
            f"({arrow}{mc['change']:.1f} {mc['unit']})"
        )
    parts.append("")

    # Normative context
    parts.append("Normative ranges (healthy elite pitchers):")
    for key, norm in NORMS.items():
        lo, hi = norm["healthy"]
        parts.append(f"  {norm['label']}: {lo}-{hi} {norm['unit']}")
    parts.append("")

    # Most divergent moment
    if comparison.frame_distances is not None:
        peak = int(np.argmax(comparison.frame_distances))
        pct = peak / len(comparison.frame_distances) * 100
        phase = "wind-up" if pct < 33 else "arm cocking" if pct < 66 else "acceleration"
        parts.append(
            f"Most divergent moment: {pct:.0f}% through delivery ({phase}), "
            f"{comparison.frame_distances[peak]:.3f}m avg displacement"
        )
        parts.append("")

    # Statcast
    if statcast:
        parts.append("Statcast performance context:")
        for key, val in statcast.items():
            parts.append(f"  {key}: {val}")
        parts.append("")

    parts.append(
        "Write your assessment. 4-6 paragraphs of narrative, "
        "then a numbered list of 3-5 actionable recommendations."
    )

    return "\n".join(parts)


def _chain_str(features) -> str:
    kc = features.kinetic_chain
    chain = [
        (kc.pelvis_peak_frame, "pelvis"),
        (kc.trunk_peak_frame, "trunk"),
        (kc.shoulder_peak_frame, "shoulder"),
        (kc.elbow_peak_frame, "elbow"),
        (kc.wrist_peak_frame, "wrist"),
    ]
    return " → ".join(name for _, name in sorted(chain))


def _parse_response(text: str) -> tuple[str, list[str]]:
    """Split LLM response into narrative and recommendations."""
    lines = text.strip().split("\n")
    narrative_lines = []
    recs = []
    in_recs = False

    for line in lines:
        stripped = line.strip()
        # Detect start of numbered recommendations
        if stripped and stripped[0].isdigit() and ("." in stripped[:3] or ")" in stripped[:3]):
            in_recs = True
            rec_text = stripped.lstrip("0123456789.-) ").strip()
            if rec_text:
                recs.append(rec_text)
        elif in_recs and stripped:
            # Continuation of recommendations section
            if stripped[0].isdigit():
                rec_text = stripped.lstrip("0123456789.-) ").strip()
                if rec_text:
                    recs.append(rec_text)
            else:
                recs.append(stripped)
        else:
            narrative_lines.append(line)

    narrative = "\n".join(narrative_lines).strip()
    return narrative, recs[:5]
