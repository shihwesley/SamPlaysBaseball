"""PDF export for ScoutingReport using reportlab."""

from __future__ import annotations

from pathlib import Path


def export_pdf(report, output_path: str) -> str:
    """Export a ScoutingReport to PDF. Returns output_path."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(str(path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    pitcher_label = report.pitcher_name or report.pitcher_id
    title_text = f"{pitcher_label} — {report.report_type.replace('_', ' ').title()} Scouting Report"
    story.append(Paragraph(title_text, styles["Title"]))
    story.append(Spacer(1, 0.1 * inch))

    # Date + risk level
    date_str = report.generated_at.strftime("%Y-%m-%d %H:%M UTC")
    story.append(Paragraph(f"Generated: {date_str} | Risk: {report.risk_level.upper()}", styles["Normal"]))
    story.append(Spacer(1, 0.2 * inch))

    # Narrative
    if report.narrative:
        story.append(Paragraph("Assessment", styles["Heading2"]))
        for para in report.narrative.split("\n\n"):
            if para.strip():
                story.append(Paragraph(para.strip(), styles["Normal"]))
                story.append(Spacer(1, 0.05 * inch))
        story.append(Spacer(1, 0.1 * inch))

    # Sections
    for section_name, section_text in report.sections.items():
        heading = section_name.replace("_", " ").title()
        story.append(Paragraph(heading, styles["Heading3"]))
        story.append(Paragraph(section_text, styles["Normal"]))
        story.append(Spacer(1, 0.08 * inch))

    # Recommendations
    if report.recommendations:
        story.append(Spacer(1, 0.1 * inch))
        story.append(Paragraph("Recommendations", styles["Heading2"]))
        for rec in report.recommendations:
            story.append(Paragraph(f"• {rec}", styles["Normal"]))
            story.append(Spacer(1, 0.04 * inch))

    doc.build(story)
    return output_path
