"""Joint index mappings for MHR70 and MHR127 skeletons."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# MHR70 key joint indices
# ---------------------------------------------------------------------------
# These are the regressed keypoints from pred_keypoints_3d (T, 70, 3).
# Best for visualization and biomechanical landmark extraction.

MHR70 = {
    # Torso
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_hip": 9,
    "right_hip": 10,
    "left_knee": 11,
    "right_knee": 12,
    "left_ankle": 13,
    "right_ankle": 14,
    # Wrists (within hand keypoint range)
    "right_wrist": 41,
    "left_wrist": 62,
    # Hand keypoints: right = 21-41, left = 42-62
    "right_hand_start": 21,
    "right_hand_end": 41,
    "left_hand_start": 42,
    "left_hand_end": 62,
}

# Inverse: index → name (for the named joints only)
MHR70_INV: dict[int, str] = {v: k for k, v in MHR70.items() if isinstance(v, int)}

# ---------------------------------------------------------------------------
# Throwing-hand aliases (right-handed pitcher defaults)
# ---------------------------------------------------------------------------

def get_throwing_joints(handedness: str = "right") -> dict[str, int]:
    """Return joint index map for the throwing arm.

    Args:
        handedness: "right" or "left"

    Returns:
        Dict mapping biomechanical role → MHR70 index.
    """
    if handedness == "right":
        return {
            "throw_shoulder": MHR70["right_shoulder"],
            "throw_elbow": MHR70["right_elbow"],
            "throw_wrist": MHR70["right_wrist"],
            "glove_shoulder": MHR70["left_shoulder"],
            "glove_elbow": MHR70["left_elbow"],
            "throw_hip": MHR70["right_hip"],
            "glove_hip": MHR70["left_hip"],
            "stride_knee": MHR70["left_knee"],
            "stride_ankle": MHR70["left_ankle"],
            "pivot_ankle": MHR70["right_ankle"],
        }
    else:
        return {
            "throw_shoulder": MHR70["left_shoulder"],
            "throw_elbow": MHR70["left_elbow"],
            "throw_wrist": MHR70["left_wrist"],
            "glove_shoulder": MHR70["right_shoulder"],
            "glove_elbow": MHR70["right_elbow"],
            "throw_hip": MHR70["left_hip"],
            "glove_hip": MHR70["right_hip"],
            "stride_knee": MHR70["right_knee"],
            "stride_ankle": MHR70["right_ankle"],
            "pivot_ankle": MHR70["left_ankle"],
        }
