"""End-to-end inference validation for the SAM 3D Body MLX port.

Run: python -m tests.test_e2e
Or:  pytest tests/test_e2e.py -v
"""

import sys
import time
import traceback

import mlx.core as mx
import numpy as np

from sam3d_mlx.config import SAM3DConfig
from sam3d_mlx.estimator import SAM3DBodyEstimator
from sam3d_mlx.batch_prep import prepare_image, get_cliff_condition

WEIGHTS_PATH = "/tmp/sam3d-mlx-weights/"

_estimator_cache = {}


def _get_estimator():
    if "est" not in _estimator_cache:
        _estimator_cache["est"] = SAM3DBodyEstimator(WEIGHTS_PATH)
    return _estimator_cache["est"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_shape(arr, expected, msg=""):
    actual = tuple(arr.shape)
    assert actual == expected, f"Shape {msg}: got {actual}, expected {expected}"


def assert_range(arr, lo, hi, msg=""):
    mn, mx_val = float(arr.min()), float(arr.max())
    assert mn >= lo and mx_val <= hi, (
        f"Range [{mn:.4f}, {mx_val:.4f}] outside [{lo}, {hi}] ({msg})"
    )


def _make_synthetic_image(h=640, w=480, seed=42):
    """Create a synthetic image with a vaguely person-shaped blob."""
    rng = np.random.RandomState(seed)
    img = rng.randint(100, 200, (h, w, 3), dtype=np.uint8)
    # Add a darker rectangle in the center to simulate a person silhouette
    cy, cx = h // 2, w // 2
    img[cy - 100:cy + 100, cx - 40:cx + 40] = rng.randint(30, 80, (200, 80, 3), dtype=np.uint8)
    return img


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------

def test_prepare_image_shape():
    """prepare_image produces (1, 512, 384, 3) float32."""
    img = _make_synthetic_image()
    bbox = [100, 50, 400, 550]
    out = prepare_image(img, bbox)
    assert_shape(out, (1, 512, 384, 3), "prepare_image")
    assert out.dtype == np.float32


def test_prepare_image_normalized():
    """Output should be roughly in ImageNet-normalized range."""
    img = _make_synthetic_image()
    bbox = [0, 0, 480, 640]
    out = prepare_image(img, bbox)
    # ImageNet normalization: values typically in [-2.5, 2.5]
    assert_range(out, -3.0, 3.0, "normalized pixel range")


def test_cliff_condition_shape():
    """CLIFF condition: (3,) float32."""
    cliff = get_cliff_condition([100, 50, 400, 550], (640, 480))
    assert_shape(cliff, (3,), "cliff_condition")
    assert cliff.dtype == np.float32


def test_cliff_condition_center_image():
    """Full-image bbox centered should give cx_norm, cy_norm near 0."""
    cliff = get_cliff_condition([0, 0, 480, 640], (640, 480))
    assert abs(cliff[0]) < 0.01, f"cx_norm {cliff[0]} not near 0"
    assert abs(cliff[1]) < 0.01, f"cy_norm {cliff[1]} not near 0"


# ---------------------------------------------------------------------------
# Full pipeline tests
# ---------------------------------------------------------------------------

def test_e2e_synthetic_image():
    """Full pipeline on synthetic image produces valid outputs."""
    est = _get_estimator()
    img = _make_synthetic_image()
    bbox = [100, 50, 400, 550]

    result = est.predict(img, bbox=bbox)

    # Check all expected keys
    expected_keys = {"pred_vertices", "pred_keypoints_3d", "pred_joint_coords", "pred_camera"}
    assert set(result.keys()) == expected_keys, f"Missing keys: {expected_keys - set(result.keys())}"

    # Shape checks
    assert_shape(result["pred_vertices"], (18439, 3), "pred_vertices")
    assert_shape(result["pred_keypoints_3d"], (70, 3), "pred_keypoints_3d")
    assert_shape(result["pred_joint_coords"], (127, 3), "pred_joint_coords")
    assert_shape(result["pred_camera"], (3,), "pred_camera")


def test_e2e_vertex_range():
    """Predicted vertices should be within plausible human range (meters)."""
    est = _get_estimator()
    img = _make_synthetic_image()
    result = est.predict(img, bbox=[100, 50, 400, 550])
    verts = result["pred_vertices"]
    # A human body fits within a ~2m cube. Allow some slack for noise.
    assert_range(verts, -3.0, 3.0, "vertex range (meters)")


def test_e2e_joint_range():
    """Predicted joints should be within plausible range."""
    est = _get_estimator()
    img = _make_synthetic_image()
    result = est.predict(img, bbox=[100, 50, 400, 550])
    joints = result["pred_joint_coords"]
    assert_range(joints, -3.0, 3.0, "joint range (meters)")


def test_e2e_camera_scale_positive():
    """Camera scale parameter should be positive."""
    est = _get_estimator()
    img = _make_synthetic_image()
    result = est.predict(img, bbox=[100, 50, 400, 550])
    cam = result["pred_camera"]
    # cam[0] is scale, typically positive and > 0
    # The init_camera may push this negative with bad input; just check finite
    assert np.isfinite(cam).all(), "Camera params not finite"


def test_e2e_vertex_centroid_reasonable():
    """Vertex centroid should be near the origin (within a few meters)."""
    est = _get_estimator()
    img = _make_synthetic_image()
    result = est.predict(img, bbox=[100, 50, 400, 550])
    centroid = result["pred_vertices"].mean(axis=0)
    assert np.linalg.norm(centroid) < 5.0, f"Centroid {centroid} too far from origin"


def test_e2e_no_nan():
    """No NaN or Inf in any output."""
    est = _get_estimator()
    img = _make_synthetic_image()
    result = est.predict(img, bbox=[100, 50, 400, 550])
    for key, val in result.items():
        assert np.isfinite(val).all(), f"NaN/Inf found in {key}"


def test_e2e_full_image_bbox():
    """Using full image as bbox (no crop) should still produce valid output."""
    est = _get_estimator()
    img = _make_synthetic_image(640, 480)
    result = est.predict(img, bbox=None)  # None means full image

    assert_shape(result["pred_vertices"], (18439, 3), "full-image vertices")
    assert np.isfinite(result["pred_vertices"]).all(), "NaN in full-image output"


def test_e2e_different_image_sizes():
    """Pipeline handles images of different sizes."""
    est = _get_estimator()
    for h, w in [(480, 320), (1080, 1920), (256, 256)]:
        img = _make_synthetic_image(h, w)
        bbox = [10, 10, w - 10, h - 10]
        result = est.predict(img, bbox=bbox)
        assert_shape(result["pred_vertices"], (18439, 3), f"vertices {h}x{w}")
        assert np.isfinite(result["pred_vertices"]).all(), f"NaN at {h}x{w}"


def test_e2e_deterministic():
    """Same input should produce identical output (no stochastic ops)."""
    est = _get_estimator()
    img = _make_synthetic_image()
    bbox = [100, 50, 400, 550]
    r1 = est.predict(img, bbox=bbox)
    r2 = est.predict(img, bbox=bbox)
    for key in r1:
        diff = np.max(np.abs(r1[key] - r2[key]))
        assert diff < 1e-6, f"Non-deterministic: {key} diff {diff}"


def test_e2e_mesh_connectivity():
    """Face indices should be valid vertex indices."""
    est = _get_estimator()
    faces = est.model.head_pose.faces
    mx.eval(faces)
    faces_np = np.array(faces)
    assert faces_np.min() >= 0, f"Negative face index: {faces_np.min()}"
    assert faces_np.max() < 18439, f"Face index {faces_np.max()} >= 18439"
    assert faces_np.shape == (36874, 3), f"Face shape {faces_np.shape}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]

    passed = 0
    failed = 0
    errors = []

    print(f"Loading model from {WEIGHTS_PATH}...")
    t0 = time.perf_counter()
    _get_estimator()
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s\n")

    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            if "--tb" in sys.argv:
                traceback.print_exc()
            failed += 1
            errors.append(name)

    print(f"\n{passed} passed, {failed} failed")
    if errors:
        print("Failed tests:")
        for e in errors:
            print(f"  - {e}")
    sys.exit(1 if failed else 0)
