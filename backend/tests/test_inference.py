"""Tests for inference, smoothing, camera, and GPU utilities.

No GPU or real model weights required — SAM 3D Body imports are not triggered
because load() is never called in these tests.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# SAM3DInference — basic instantiation and guard
# ---------------------------------------------------------------------------

def test_sam3d_inference_instantiates_without_loading():
    # Importing inference.py does sys.path.insert but does NOT import sam_3d_body
    # (that only happens inside load()). So this must succeed even without /tmp/sam-3d-body.
    from backend.app.pipeline.inference import SAM3DInference
    infer = SAM3DInference(device="cpu")
    assert not infer._loaded


def test_process_frame_raises_if_not_loaded():
    from backend.app.pipeline.inference import SAM3DInference
    infer = SAM3DInference(device="cpu")
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="load()"):
        infer.process_frame(frame)


# ---------------------------------------------------------------------------
# smooth_joints
# ---------------------------------------------------------------------------

def _random_joints(T: int, J: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal((T, J, 3)).astype(np.float32)


def test_smooth_joints_none_returns_identical():
    from backend.app.pipeline.smoothing import smooth_joints
    joints = _random_joints(20, 10)
    result = smooth_joints(joints, method="none")
    np.testing.assert_array_equal(result, joints)


def test_smooth_joints_butterworth_same_shape():
    from backend.app.pipeline.smoothing import smooth_joints
    joints = _random_joints(30, 5)
    result = smooth_joints(joints, method="butterworth")
    assert result.shape == joints.shape


def test_smooth_joints_kalman_same_shape():
    from backend.app.pipeline.smoothing import smooth_joints
    joints = _random_joints(10, 5)
    result = smooth_joints(joints, method="kalman")
    assert result.shape == joints.shape


def test_smooth_joints_invalid_method_raises():
    from backend.app.pipeline.smoothing import smooth_joints
    with pytest.raises(ValueError, match="Unknown smoothing method"):
        smooth_joints(_random_joints(10, 3), method="invalid")


# ---------------------------------------------------------------------------
# camera
# ---------------------------------------------------------------------------

def test_estimate_camera_focal_length_1920x1080():
    from backend.app.pipeline.camera import estimate_camera
    params = estimate_camera(frame_width=1920, frame_height=1080)
    assert params.focal_length == 1920.0
    assert params.cx == 960.0
    assert params.cy == 540.0


def test_estimate_camera_focal_length_tall_frame():
    from backend.app.pipeline.camera import estimate_camera
    params = estimate_camera(frame_width=720, frame_height=1280)
    assert params.focal_length == 1280.0


def test_override_camera_focal_length():
    from backend.app.pipeline.camera import estimate_camera, override_camera
    base = estimate_camera(1920, 1080)
    overridden = override_camera(base, focal_length=2500.0)
    assert overridden.focal_length == 2500.0
    assert overridden.cx == base.cx  # unchanged


def test_override_camera_cam_t():
    from backend.app.pipeline.camera import estimate_camera, override_camera
    base = estimate_camera(1280, 720)
    new_t = np.array([0.1, 0.2, 3.0])
    overridden = override_camera(base, cam_t=new_t)
    np.testing.assert_array_equal(overridden.cam_t, new_t)
    assert overridden.focal_length == base.focal_length  # unchanged


# ---------------------------------------------------------------------------
# gpu
# ---------------------------------------------------------------------------

def test_get_device_returns_torch_device():
    from backend.app.pipeline.gpu import get_device
    device = get_device()
    assert isinstance(device, torch.device)


def test_auto_batch_size_thresholds():
    from backend.app.pipeline.gpu import auto_batch_size
    assert auto_batch_size(9.0) == 4
    assert auto_batch_size(8.1) == 4
    assert auto_batch_size(8.0) == 2   # not > 8, so 2
    assert auto_batch_size(5.0) == 2
    assert auto_batch_size(4.0) == 1   # not > 4, so 1
    assert auto_batch_size(2.0) == 1


def test_memory_stats_returns_dict():
    from backend.app.pipeline.gpu import memory_stats
    stats = memory_stats()
    assert "device" in stats
    assert "available" in stats
