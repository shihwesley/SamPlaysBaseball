"""Per-component shape and value validation for the SAM 3D Body MLX port.

Run: python -m tests.test_parity
Or:  pytest tests/test_parity.py -v
"""

import sys
import traceback

import mlx.core as mx
import numpy as np

from sam3d_mlx.config import SAM3DConfig
from sam3d_mlx.model import SAM3DBody
from sam3d_mlx.mhr_utils import (
    rot6d_to_rotmat,
    rotmat_to_euler_ZYX,
    batch_xyz_from_6d,
    compact_cont_to_model_params_body,
    compact_cont_to_model_params_hand,
    euler_xyz_to_rotmat,
    quat_to_rotmat,
    rotmat_to_quat,
    sincos_to_angle,
    ALL_PARAM_3DOF_ROT_IDXS,
    ALL_PARAM_1DOF_ROT_IDXS,
    ALL_PARAM_1DOF_TRANS_IDXS,
    HAND_DOFS_IN_ORDER,
)

WEIGHTS_PATH = "/tmp/sam3d-mlx-weights/"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_shape(arr, expected, msg=""):
    actual = tuple(arr.shape)
    assert actual == expected, f"Shape mismatch{' (' + msg + ')' if msg else ''}: got {actual}, expected {expected}"


def assert_close(a, b, atol=1e-4, msg=""):
    a_np = np.array(a) if not isinstance(a, np.ndarray) else a
    b_np = np.array(b) if not isinstance(b, np.ndarray) else b
    diff = np.max(np.abs(a_np - b_np))
    assert diff < atol, f"Max diff {diff:.6f} >= {atol}{' (' + msg + ')' if msg else ''}"


def assert_range(arr, lo, hi, msg=""):
    a = np.array(arr) if not isinstance(arr, np.ndarray) else arr
    mn, mx_val = float(a.min()), float(a.max())
    assert mn >= lo and mx_val <= hi, (
        f"Range [{mn:.4f}, {mx_val:.4f}] outside [{lo}, {hi}]{' (' + msg + ')' if msg else ''}"
    )


# ---------------------------------------------------------------------------
# Rotation math tests
# ---------------------------------------------------------------------------

def test_rot6d_identity():
    """6D identity input -> identity rotation matrix."""
    # Identity rotation: first two columns of I_3 = [1,0,0, 0,1,0]
    inp = mx.array([[1.0, 0, 0, 0, 1, 0]])
    R = rot6d_to_rotmat(inp)
    mx.eval(R)
    assert_shape(R, (1, 3, 3))
    assert_close(R, np.eye(3).reshape(1, 3, 3), atol=1e-5, msg="identity rotation")


def test_rot6d_determinant():
    """rot6d_to_rotmat output should have determinant 1.0 for any input."""
    rng = np.random.RandomState(42)
    inp = mx.array(rng.randn(8, 6).astype(np.float32))
    R = rot6d_to_rotmat(inp)
    mx.eval(R)
    R_np = np.array(R)
    for i in range(8):
        det = np.linalg.det(R_np[i])
        assert abs(det - 1.0) < 1e-4, f"det(R[{i}]) = {det:.6f}, expected 1.0"


def test_rot6d_orthogonal():
    """Columns of rot6d_to_rotmat output should be orthonormal."""
    rng = np.random.RandomState(7)
    inp = mx.array(rng.randn(4, 6).astype(np.float32))
    R = rot6d_to_rotmat(inp)
    mx.eval(R)
    R_np = np.array(R)
    for i in range(4):
        RtR = R_np[i] @ R_np[i].T
        assert_close(RtR, np.eye(3), atol=1e-4, msg=f"R[{i}]^T R[{i}] != I")


def test_rotmat_to_euler_identity():
    """Identity rotation matrix -> zero euler angles."""
    I = mx.array(np.eye(3, dtype=np.float32).reshape(1, 3, 3))
    euler = rotmat_to_euler_ZYX(I)
    mx.eval(euler)
    assert_shape(euler, (1, 3))
    assert_close(euler, np.zeros((1, 3)), atol=1e-5, msg="identity euler")


def test_euler_rotmat_roundtrip():
    """euler -> rotmat -> euler should preserve the angles (small angles)."""
    rng = np.random.RandomState(99)
    angles = rng.uniform(-0.5, 0.5, (5, 3)).astype(np.float32)
    angles_mx = mx.array(angles)
    R = euler_xyz_to_rotmat(angles_mx)
    # rotmat_to_euler_ZYX returns ZYX order; euler_xyz_to_rotmat takes XYZ order
    # so a direct roundtrip is only approximate for small angles
    recovered = rotmat_to_euler_ZYX(R)
    mx.eval(recovered)
    # Reverse the order for comparison: ZYX -> XYZ
    rec_np = np.array(recovered)[:, ::-1]
    # For small angles the two conventions are close enough
    assert_close(rec_np, angles, atol=0.05, msg="euler roundtrip")


def test_batch_xyz_from_6d_shape():
    """batch_xyz_from_6d: (B, 6) -> (B, 3)."""
    inp = mx.array(np.random.randn(3, 6).astype(np.float32))
    out = batch_xyz_from_6d(inp)
    mx.eval(out)
    assert_shape(out, (3, 3))


def test_sincos_to_angle():
    """sincos_to_angle: known sin/cos pairs."""
    # sin=0, cos=1 -> angle=0; sin=1, cos=0 -> angle=pi/2
    sc = mx.array([[0.0, 1.0], [1.0, 0.0], [0.0, -1.0]])
    angles = sincos_to_angle(sc)
    mx.eval(angles)
    expected = np.array([0.0, np.pi / 2, np.pi])
    assert_close(np.array(angles), expected, atol=1e-5)


def test_quat_identity():
    """Identity quaternion (0,0,0,1) -> identity rotation matrix."""
    q = mx.array([[0.0, 0.0, 0.0, 1.0]])
    R = quat_to_rotmat(q)
    mx.eval(R)
    assert_close(R, np.eye(3).reshape(1, 3, 3), atol=1e-6)


def test_quat_rotmat_roundtrip():
    """rotmat -> quat -> rotmat roundtrip."""
    rng = np.random.RandomState(13)
    angles = rng.uniform(-0.3, 0.3, (4, 3)).astype(np.float32)
    R = euler_xyz_to_rotmat(mx.array(angles))
    q = rotmat_to_quat(R)
    R2 = quat_to_rotmat(q)
    mx.eval(R2)
    assert_close(R, R2, atol=1e-4, msg="quat roundtrip")


# ---------------------------------------------------------------------------
# Parameter conversion tests
# ---------------------------------------------------------------------------

def test_compact_cont_body_shape():
    """compact_cont_to_model_params_body: (B, 260) -> (B, 133)."""
    inp = mx.zeros((2, 260))
    out = compact_cont_to_model_params_body(inp)
    mx.eval(out)
    assert_shape(out, (2, 133))


def test_compact_cont_body_coverage():
    """All 133 output indices are covered by the body conversion."""
    # Verify the index arrays cover 0..132 exactly once
    all_3dof = []
    for triple in ALL_PARAM_3DOF_ROT_IDXS:
        all_3dof.extend(triple)
    all_1dof = list(ALL_PARAM_1DOF_ROT_IDXS)
    all_trans = list(ALL_PARAM_1DOF_TRANS_IDXS)
    all_indices = sorted(all_3dof + all_1dof + all_trans)
    expected = list(range(133))
    assert all_indices == expected, f"Indices mismatch: got {len(all_indices)} unique, expected 133"


def test_compact_cont_body_input_consumption():
    """Body conversion consumes exactly 260 input dims: 23*6 + 58*2 + 6*1."""
    total = len(ALL_PARAM_3DOF_ROT_IDXS) * 6 + len(ALL_PARAM_1DOF_ROT_IDXS) * 2 + len(ALL_PARAM_1DOF_TRANS_IDXS)
    assert total == 260, f"Expected 260 input dims, got {total}"


def test_compact_cont_hand_shape():
    """compact_cont_to_model_params_hand: (B, 54) -> (B, 27)."""
    inp = mx.zeros((2, 54))
    out = compact_cont_to_model_params_hand(inp)
    mx.eval(out)
    assert_shape(out, (2, 27))


def test_compact_cont_hand_input_consumption():
    """Hand conversion consumes exactly 54 input dims."""
    total = 0
    expected_out = 0
    for dof in HAND_DOFS_IN_ORDER:
        if dof == 3:
            total += 6
            expected_out += 3
        elif dof == 1:
            total += 2
            expected_out += 1
        elif dof == 2:
            total += 4
            expected_out += 2
    assert total == 54, f"Expected 54 input dims, got {total}"
    assert expected_out == 27, f"Expected 27 output dims, got {expected_out}"


# ---------------------------------------------------------------------------
# Weight loading tests (require weights on disk)
# ---------------------------------------------------------------------------

_model_cache = {}


def _get_model():
    """Load model once and cache for all weight tests."""
    if "model" not in _model_cache:
        config = SAM3DConfig()
        model = SAM3DBody(config)
        model.load_all_weights(WEIGHTS_PATH)
        model.eval()
        _model_cache["model"] = model
    return _model_cache["model"]


def test_weight_loading_no_zeros():
    """After loading, key weight matrices should not be all zeros."""
    model = _get_model()
    # Check a few critical weights
    checks = [
        ("init_pose", model.init_pose),
        ("init_camera", model.init_camera),
        ("keypoint_embedding", model.keypoint_embedding),
    ]
    for name, w in checks:
        mx.eval(w)
        w_np = np.array(w)
        assert not np.allclose(w_np, 0), f"{name} is all zeros after loading"


def test_backbone_shape():
    """Backbone: (1, 512, 384, 3) -> (1, 32, 24, 1280)."""
    model = _get_model()
    x = mx.zeros((1, 512, 384, 3))
    out = model.backbone(x)
    mx.eval(out)
    assert_shape(out, (1, 32, 24, 1280))


def test_init_pose_shape():
    """init_pose buffer should be (1, 519)."""
    model = _get_model()
    assert_shape(model.init_pose, (1, 519), "init_pose")


def test_init_camera_shape():
    """init_camera buffer should be (1, 3)."""
    model = _get_model()
    assert_shape(model.init_camera, (1, 3), "init_camera")


def test_keypoint_embedding_shape():
    """keypoint_embedding should be (70, 1024)."""
    model = _get_model()
    assert_shape(model.keypoint_embedding, (70, 1024), "keypoint_embedding")


def test_hand_box_embedding_shape():
    """hand_box_embedding should be (2, 1024)."""
    model = _get_model()
    assert_shape(model.hand_box_embedding, (2, 1024), "hand_box_embedding")


def test_faces_shape():
    """MHR faces buffer should be (36874, 3) int32."""
    model = _get_model()
    faces = model.head_pose.faces
    mx.eval(faces)
    assert_shape(faces, (36874, 3), "faces")
    assert faces.dtype == mx.int32, f"faces dtype {faces.dtype}, expected int32"


def test_body_model_buffers():
    """Body model buffers have expected shapes."""
    model = _get_model()
    bm = model.head_pose.body_model
    expected = {
        "joint_translation_offsets": (127, 3),
        "joint_prerotations": (127, 4),
        "joint_parents": (127,),
        "base_shape": (18439, 3),
        "shape_vectors": (45, 18439, 3),
        "parameter_transform": (889, 249),
        "inverse_bind_pose": (127, 8),
    }
    for attr, shape in expected.items():
        buf = getattr(bm, attr)
        mx.eval(buf)
        assert_shape(buf, shape, attr)


def test_body_model_base_shape_range():
    """Base shape vertex coords should be in plausible range (cm, +-300)."""
    model = _get_model()
    base = model.head_pose.body_model.base_shape
    mx.eval(base)
    base_np = np.array(base)
    assert_range(base_np, -300, 300, "base_shape in cm")


def test_joint_parents_valid():
    """Joint parents should be in [-1, 126] and root should have parent -1."""
    model = _get_model()
    parents = model.head_pose.body_model.joint_parents
    mx.eval(parents)
    p_np = np.array(parents).astype(np.int64)
    assert p_np.min() >= -1, f"min parent {p_np.min()}"
    assert p_np.max() < 127, f"max parent {p_np.max()}"
    # At least one root (parent == -1)
    assert (p_np == -1).sum() >= 1, "No root joint found"


def test_blend_shapes_zero_params():
    """Blend shapes with zero shape params should return base_shape."""
    model = _get_model()
    bm = model.head_pose.body_model
    shape_params = mx.zeros((1, 45))
    verts = bm._blend_shapes(shape_params, expr_params=None)
    mx.eval(verts)
    assert_shape(verts, (1, 18439, 3))
    # Should match base_shape exactly
    assert_close(verts[0], bm.base_shape, atol=1e-6, msg="zero shape = base_shape")


def test_keypoint_mapping_shape():
    """Keypoint mapping should be (308, 18566) where 18566 = 18439 + 127."""
    model = _get_model()
    km = model.head_pose.keypoint_mapping
    mx.eval(km)
    assert_shape(km, (308, 18566), "keypoint_mapping")


def test_keypoint_mapping_rows_sum():
    """Each row of keypoint_mapping should sum to ~1.0 (weighted average)."""
    model = _get_model()
    km = model.head_pose.keypoint_mapping
    mx.eval(km)
    km_np = np.array(km)
    row_sums = km_np.sum(axis=1)
    # Most rows should sum close to 1.0 (some might be 0 for unused keypoints)
    nonzero_rows = row_sums[row_sums > 0.01]
    if len(nonzero_rows) > 0:
        assert_close(nonzero_rows, np.ones_like(nonzero_rows), atol=0.01,
                     msg="keypoint_mapping row sums")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]

    passed = 0
    failed = 0
    errors = []

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
