"""Joint angle computation from 3D keypoint coordinates."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def angle_between(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    c: NDArray[np.float64],
) -> float:
    """Compute angle at vertex b formed by points a-b-c.

    Uses arccos of the dot product of unit vectors ba and bc.
    Returns angle in degrees.
    """
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-9 or norm_bc < 1e-9:
        return 0.0
    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def angle_sequence(
    joints: NDArray[np.float64],
    idx_a: int,
    idx_b: int,
    idx_c: int,
) -> NDArray[np.float64]:
    """Compute joint angle at b for every frame.

    Args:
        joints: (T, J, 3) array of joint positions.
        idx_a: proximal joint index.
        idx_b: vertex joint index.
        idx_c: distal joint index.

    Returns:
        (T,) array of angles in degrees.
    """
    T = joints.shape[0]
    angles = np.zeros(T)
    for t in range(T):
        angles[t] = angle_between(joints[t, idx_a], joints[t, idx_b], joints[t, idx_c])
    return angles


def elbow_flexion(
    joints: NDArray[np.float64],
    shoulder_idx: int,
    elbow_idx: int,
    wrist_idx: int,
) -> NDArray[np.float64]:
    """Elbow flexion angle (shoulder-elbow-wrist), degrees."""
    return angle_sequence(joints, shoulder_idx, elbow_idx, wrist_idx)


def shoulder_abduction(
    joints: NDArray[np.float64],
    elbow_idx: int,
    shoulder_idx: int,
    contra_shoulder_idx: int,
) -> NDArray[np.float64]:
    """Shoulder abduction: angle in the frontal plane.

    Approximated as elbow-shoulder-contralateral_shoulder.
    """
    return angle_sequence(joints, elbow_idx, shoulder_idx, contra_shoulder_idx)


def hip_flexion(
    joints: NDArray[np.float64],
    knee_idx: int,
    hip_idx: int,
    contra_hip_idx: int,
) -> NDArray[np.float64]:
    """Hip flexion angle (knee-hip-contra_hip), degrees."""
    return angle_sequence(joints, knee_idx, hip_idx, contra_hip_idx)


def knee_flexion(
    joints: NDArray[np.float64],
    hip_idx: int,
    knee_idx: int,
    ankle_idx: int,
) -> NDArray[np.float64]:
    """Knee flexion angle, degrees."""
    return angle_sequence(joints, hip_idx, knee_idx, ankle_idx)


def trunk_tilt(
    joints: NDArray[np.float64],
    left_shoulder_idx: int,
    right_shoulder_idx: int,
    left_hip_idx: int,
    right_hip_idx: int,
) -> NDArray[np.float64]:
    """Lateral trunk tilt: angle between shoulder midpoint vertical and world vertical.

    Returns degrees of lateral lean (positive = glove-side lean).
    """
    T = joints.shape[0]
    tilts = np.zeros(T)
    world_up = np.array([0.0, 1.0, 0.0])
    for t in range(T):
        shoulder_mid = (joints[t, left_shoulder_idx] + joints[t, right_shoulder_idx]) / 2
        hip_mid = (joints[t, left_hip_idx] + joints[t, right_hip_idx]) / 2
        trunk_vec = shoulder_mid - hip_mid
        norm = np.linalg.norm(trunk_vec)
        if norm < 1e-9:
            continue
        trunk_unit = trunk_vec / norm
        cos_a = np.clip(np.dot(trunk_unit, world_up), -1.0, 1.0)
        tilts[t] = float(np.degrees(np.arccos(cos_a)))
    return tilts


def trunk_rotation(
    joints: NDArray[np.float64],
    left_shoulder_idx: int,
    right_shoulder_idx: int,
    left_hip_idx: int,
    right_hip_idx: int,
) -> NDArray[np.float64]:
    """Trunk rotation: angle between shoulder line and hip line projected onto horizontal plane.

    Returns signed degrees. Positive = throwing-side open.
    """
    T = joints.shape[0]
    rotations = np.zeros(T)
    for t in range(T):
        shoulder_vec = joints[t, right_shoulder_idx] - joints[t, left_shoulder_idx]
        hip_vec = joints[t, right_hip_idx] - joints[t, left_hip_idx]
        # project onto horizontal plane (zero out y)
        sv = np.array([shoulder_vec[0], 0.0, shoulder_vec[2]])
        hv = np.array([hip_vec[0], 0.0, hip_vec[2]])
        ns = np.linalg.norm(sv)
        nh = np.linalg.norm(hv)
        if ns < 1e-9 or nh < 1e-9:
            continue
        cos_a = np.clip(np.dot(sv / ns, hv / nh), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_a))
        # sign: cross product y-component
        cross = np.cross(sv / ns, hv / nh)
        rotations[t] = angle if cross[1] >= 0 else -angle
    return rotations


def shoulder_external_rotation(
    joints: NDArray[np.float64],
    elbow_idx: int,
    shoulder_idx: int,
    contra_shoulder_idx: int,
    wrist_idx: int,
) -> NDArray[np.float64]:
    """Approximate shoulder external rotation.

    Projects the forearm vector onto the plane perpendicular to the upper-arm axis
    and measures its angle relative to the horizontal reference in that plane.

    Returns (T,) degrees, where larger = more external rotation (normative peak ~170 deg).
    """
    T = joints.shape[0]
    er = np.zeros(T)
    for t in range(T):
        upper_arm = joints[t, elbow_idx] - joints[t, shoulder_idx]
        forearm = joints[t, wrist_idx] - joints[t, elbow_idx]
        ua_norm = np.linalg.norm(upper_arm)
        if ua_norm < 1e-9:
            continue
        ua_unit = upper_arm / ua_norm
        # project forearm onto plane perpendicular to upper arm
        forearm_proj = forearm - np.dot(forearm, ua_unit) * ua_unit
        # reference: horizontal direction perpendicular to upper arm
        ref = joints[t, contra_shoulder_idx] - joints[t, shoulder_idx]
        ref_proj = ref - np.dot(ref, ua_unit) * ua_unit
        fn = np.linalg.norm(forearm_proj)
        rn = np.linalg.norm(ref_proj)
        if fn < 1e-9 or rn < 1e-9:
            continue
        cos_a = np.clip(np.dot(forearm_proj / fn, ref_proj / rn), -1.0, 1.0)
        er[t] = float(np.degrees(np.arccos(cos_a)))
    return er
