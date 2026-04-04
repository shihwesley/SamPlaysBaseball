"""LLM integration for scouting report narrative generation."""

from __future__ import annotations

from anthropic import Anthropic

_SYSTEM_PROMPT = (
    "You are a professional baseball scout writing an internal assessment. "
    "Write factual, specific, concise reports. Use scout terminology. No filler phrases."
)


class LLMReportGenerator:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-opus-4-5",
    ) -> None:
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def generate_narrative(
        self, sections: dict[str, str], pitcher_name: str
    ) -> str:
        """Assemble section summaries into a coherent scouting narrative.

        sections: dict of section_name -> section_text
        Returns: multi-paragraph narrative string
        """
        section_block = "\n\n".join(
            f"[{name.upper()}]\n{text}" for name, text in sections.items()
        )
        user_message = (
            f"Write a concise internal scouting report for {pitcher_name}. "
            "Use the following analysis sections. Integrate the data into a coherent, "
            "professional assessment. 3-4 paragraphs. Specific numbers, scout language:\n\n"
            f"{section_block}"
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=800,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text.strip()

    def generate_recommendations(
        self, risk_factors: dict, analysis_summary: dict
    ) -> list[str]:
        """Return 3-5 actionable recommendations in scout language."""
        risk_str = "\n".join(
            f"- {k}: {v:.3f}" if isinstance(v, float) else f"- {k}: {v}"
            for k, v in risk_factors.items()
        )
        summary_str = "\n".join(
            f"- {k}: {v}" for k, v in analysis_summary.items()
        )
        user_message = (
            "Based on the following biomechanical risk factors and analysis summary, "
            "provide 3-5 actionable recommendations. Each recommendation should be one "
            "specific, concrete action. Scout/coach language. No filler.\n\n"
            f"Risk factors:\n{risk_str}\n\nAnalysis summary:\n{summary_str}"
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=400,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        text = response.content[0].text.strip()
        # Split on numbered list markers or newlines
        lines = [
            line.lstrip("0123456789.-) ").strip()
            for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        return [l for l in lines if l][:5]
