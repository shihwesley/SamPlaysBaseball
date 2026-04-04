"""Report section templates — factual, scout-language summaries."""

from __future__ import annotations


def pitcher_profile_section(
    pitcher_id: str,
    name: str | None,
    pitch_types: list[str],
    sample_counts: dict[str, int],
) -> str:
    display = name or pitcher_id
    pt_list = ", ".join(
        f"{pt} ({sample_counts.get(pt, 0)} pitches)" for pt in pitch_types
    ) if pitch_types else "no pitch types on record"
    return (
        f"{display} — arsenal: {pt_list}. "
        f"Total tracked appearances: {sum(sample_counts.values())} pitches across {len(pitch_types)} pitch types."
    )


def tipping_section(
    tip_signals: list[dict],
    max_score: float,
    is_tipping: bool,
) -> str:
    if not is_tipping or not tip_signals:
        return "No significant pitch-tipping signals detected. Mechanical consistency across pitch types is within acceptable variance."
    top = sorted(tip_signals, key=lambda s: s.get("separation_score", 0), reverse=True)[:2]
    signal_strs = []
    for sig in top:
        a = sig.get("pitch_type_a", "?")
        b = sig.get("pitch_type_b", "?")
        feat = sig.get("feature_name", "unknown feature")
        score = sig.get("separation_score", 0)
        signal_strs.append(f"{feat} separates {a}/{b} at {score:.2f} separation score")
    return (
        f"Tipping risk detected (max separation score {max_score:.2f}). "
        f"Primary signals: {'; '.join(signal_strs)}. "
        "Hitters with strong pre-pitch read ability may exploit these tells."
    )


def fatigue_section(
    markers: list[dict],
    fatigue_score: float,
    is_fatigued: bool,
) -> str:
    level = "elevated" if is_fatigued else "acceptable"
    marker_str = ""
    if markers:
        top = markers[:2]
        parts = []
        for m in top:
            name = m.get("metric_name", "metric")
            pct = m.get("pct_change", 0)
            parts.append(f"{name} {pct:+.1f}%")
        marker_str = f" Key drift: {', '.join(parts)}."
    return (
        f"Fatigue score {fatigue_score:.2f}/1.00 — {level}.{marker_str} "
        f"{'Mechanical breakdown consistent with cumulative load.' if is_fatigued else 'No significant load-related decline.'}"
    )


def command_section(
    command_score: float,
    release_deviations: dict | None,
) -> str:
    grade = "strong" if command_score >= 0.75 else ("average" if command_score >= 0.50 else "below average")
    dev_str = ""
    if release_deviations:
        rx = release_deviations.get("release_x_deviation")
        rz = release_deviations.get("release_z_deviation")
        if rx is not None and rz is not None:
            dev_str = f" Release point deviation: {rx:.1f}mm horizontal, {rz:.1f}mm vertical."
    return (
        f"Command score {command_score:.2f} — {grade}.{dev_str} "
        "Reflects release point consistency relative to pitcher baseline."
    )


def arm_slot_section(
    drift_degrees: float | None,
    baseline_degrees: float | None,
    is_significant: bool,
) -> str:
    if drift_degrees is None:
        return "Arm slot drift data unavailable for this sample."
    drift_str = f"{drift_degrees:.1f}° drift"
    baseline_str = f" (baseline {baseline_degrees:.1f}°)" if baseline_degrees is not None else ""
    flag = " Significant slot instability — review release mechanics." if is_significant else " Within normal variance."
    return f"Arm slot: {drift_str}{baseline_str}.{flag}"


def timing_section(
    events: list[dict],
    timing_score: float,
    is_timing_issue: bool,
) -> str:
    flag = "timing disruption present" if is_timing_issue else "timing within normal range"
    event_str = ""
    if events:
        late = [e for e in events if (e.get("frame_delta") or 0) > 0]
        early = [e for e in events if (e.get("frame_delta") or 0) < 0]
        parts = []
        if late:
            parts.append(f"{len(late)} event(s) late")
        if early:
            parts.append(f"{len(early)} event(s) early")
        if parts:
            event_str = f" {', '.join(parts)} vs. baseline."
    return (
        f"Timing score {timing_score:.2f} — {flag}.{event_str} "
        "Kinetic chain sequencing assessed relative to pitcher's own baseline delivery."
    )


def injury_risk_section(
    risk_score: float,
    traffic_light: str,
    top_factors: list[tuple[str, float]],
) -> str:
    factor_str = ""
    if top_factors:
        parts = [
            f"{name.replace('_', ' ')} ({val:.2f})"
            for name, val in top_factors[:3]
        ]
        factor_str = f" Top drivers: {', '.join(parts)}."
    return (
        f"Biomechanical load indicator: {risk_score:.1f}/100 ({traffic_light.upper()}).{factor_str} "
        "Score reflects ASMI-weighted mechanical load correlates — not injury prediction."
    )


def statcast_section(correlation_data: dict | None) -> str:
    if not correlation_data:
        return "Statcast correlation data not available for this sample."
    corr = correlation_data.get("correlation_summary", {})
    parts = []
    for metric, val in list(corr.items())[:3]:
        if isinstance(val, (int, float)):
            parts.append(f"{metric}: r={val:.2f}")
    if not parts:
        return "Statcast correlation computed but summary unavailable."
    return (
        f"Statcast correlations: {'; '.join(parts)}. "
        "Mechanical markers compared against Statcast velocity, spin rate, and zone outcomes."
    )
