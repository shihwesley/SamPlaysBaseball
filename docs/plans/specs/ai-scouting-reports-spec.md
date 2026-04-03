---
name: ai-scouting-reports
phase: 3
sprint: 1
parent: null
depends_on: [baseline-comparison, tipping-detection, fatigue-tracking, command-analysis, arm-slot-drift, timing-analysis, injury-risk, statcast-integration]
status: draft
created: 2026-02-16
---

# AI-Generated Scouting Reports Spec

LLM-powered one-page reports that translate raw analysis into the language scouts and player dev staff actually use.

## Requirements

- Generate human-readable summaries from all analysis module outputs
- Match the format and tone of professional scouting reports
- Actionable recommendations, not just data dumps
- One-page per-pitcher report + per-outing report

## Acceptance Criteria

- [ ] Per-pitcher scouting report: one page, covers all pitch types, mechanical strengths/weaknesses, risk factors
- [ ] Per-outing report: how did this start go, when did mechanics change, what happened
- [ ] Per-pitch-type breakdown: "Fastball: 94.2mph avg, tight release, good arm slot. Slider: release point drifts 1.8cm arm-side vs FB — potential tipping concern."
- [ ] Actionable recommendations: "Focus on maintaining glove height during set position to eliminate changeup tell"
- [ ] Injury risk summary in plain English: "Fatigue pattern starting at pitch 78 shows arm slot drop consistent with increased elbow stress"
- [ ] Statcast integration summary: "When hip-shoulder separation exceeds 50 degrees, fastball velocity averages 1.2mph higher"
- [ ] Tone: professional, concise, factual — like a real scout wrote it
- [ ] PDF export option
- [ ] Template system: customizable report sections

## Technical Approach

Use Claude API (or any LLM) with structured prompts. Feed analysis results as structured data, prompt generates natural language report. Template-based: assemble report sections from per-module summaries, then run through LLM for coherence and tone.

Key prompt engineering: provide the LLM with scouting report examples and baseball biomechanics vocabulary. The output should read like a Driveline report or a professional scouting assessment, not like an AI summary.

Fallback: if LLM unavailable, generate template-based reports with variable substitution (less natural but still functional).

PDF generation via WeasyPrint or reportlab for the export feature.

## Files

| File | Purpose |
|------|---------|
| backend/app/reports/generator.py | ReportGenerator class |
| backend/app/reports/templates.py | Report section templates |
| backend/app/reports/llm.py | LLM integration (Claude API) |
| backend/app/reports/pdf.py | PDF export |
| backend/tests/test_reports.py | Report generation tests |

## Tasks

1. Design report template structure (sections, data requirements per section)
2. Build data assembly (gather analysis results into structured prompt input)
3. Implement LLM integration for natural language generation
4. Build template-based fallback (variable substitution without LLM)
5. Implement PDF export
6. Test with sample data and iterate on prompt quality

## Dependencies

- Upstream: all analysis modules, injury-risk, statcast-integration
- Downstream: api-layer, dashboard-ui
