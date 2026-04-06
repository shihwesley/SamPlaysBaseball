"""Normative biomechanical ranges for pitching mechanics.

Sources:
- Driveline OpenBiomechanics (100K+ pitches)
- ASMI published ranges
- Fleisig et al. (2011) kinematic parameters of elite pitchers
"""

NORMS = {
    "max_shoulder_er_deg": {
        "label": "Max Shoulder External Rotation",
        "healthy": [150, 180],
        "concern_below": 140,
        "concern_above": 190,
        "unit": "deg",
    },
    "hip_shoulder_sep_deg": {
        "label": "Hip-Shoulder Separation",
        "healthy": [35, 60],
        "concern_below": 25,
        "unit": "deg",
    },
    "stride_length_normalized": {
        "label": "Stride Length (normalized)",
        "healthy": [0.75, 0.90],
        "concern_below": 0.65,
        "unit": "x height",
    },
    "elbow_flexion_vel": {
        "label": "Peak Elbow Flexion Velocity",
        "healthy": [2000, 2800],
        "concern_above": 3200,
        "unit": "deg/s",
    },
    "shoulder_er_vel": {
        "label": "Peak Shoulder ER Velocity",
        "healthy": [800, 1500],
        "unit": "deg/s",
    },
    "trunk_rotation_range": {
        "label": "Trunk Rotation Range",
        "healthy": [25, 50],
        "concern_below": 15,
        "unit": "deg",
    },
    "elbow_flexion_range": {
        "label": "Elbow Flexion Range",
        "healthy": [60, 120],
        "unit": "deg",
    },
    "knee_flexion_at_fp": {
        "label": "Knee Flexion at Foot Plant",
        "healthy": [35, 55],
        "unit": "deg",
    },
}


def flag_concerns(mechanical_changes: list[dict]) -> list[str]:
    """Check mechanical changes against norms, return risk flags."""
    flags = []
    for change in mechanical_changes:
        name = change["name"]
        # Map display names back to norm keys
        norm_key = _display_to_key(name)
        if norm_key not in NORMS:
            continue
        norm = NORMS[norm_key]
        # Check the "late" value (current) against thresholds
        val = change.get("late") or change.get("value_b")
        if val is None:
            continue
        low = norm.get("concern_below")
        high = norm.get("concern_above")
        if low is not None and val < low:
            flags.append(f"{norm['label']} at {val:.1f}{norm['unit']} — below {low}{norm['unit']} threshold")
        if high is not None and val > high:
            flags.append(f"{norm['label']} at {val:.1f}{norm['unit']} — above {high}{norm['unit']} threshold")
    return flags


_DISPLAY_KEY_MAP = {
    "max shoulder er": "max_shoulder_er_deg",
    "hip-shoulder sep": "hip_shoulder_sep_deg",
    "stride length": "stride_length_normalized",
    "peak elbow flex vel": "elbow_flexion_vel",
    "peak shoulder er vel": "shoulder_er_vel",
    "trunk rotation range": "trunk_rotation_range",
    "elbow flex range": "elbow_flexion_range",
}


def _display_to_key(display_name: str) -> str:
    lower = display_name.lower().strip()
    for pattern, key in _DISPLAY_KEY_MAP.items():
        if pattern in lower:
            return key
    return lower.replace(" ", "_")
