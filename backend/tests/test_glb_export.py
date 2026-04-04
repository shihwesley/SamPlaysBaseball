"""Tests for the mesh export package."""
import sys
import os
import pytest
import numpy as np
import tempfile

# Ensure the worktree root is on sys.path so `backend.app.export` resolves
_WORKTREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _WORKTREE not in sys.path:
    sys.path.insert(0, _WORKTREE)

# Skip conditions
try:
    import pygltflib  # noqa: F401
    HAS_PYGLTFLIB = True
except ImportError:
    HAS_PYGLTFLIB = False

requires_pygltflib = pytest.mark.skipif(
    not HAS_PYGLTFLIB,
    reason="pygltflib not installed — run: pip install pygltflib",
)


def _make_joint_sequence(n_frames: int = 10, n_joints: int = 70) -> list[np.ndarray]:
    rng = np.random.default_rng(42)
    return [rng.random((n_joints, 3)).astype(np.float32) for _ in range(n_frames)]


def _make_mesh_frames(n_frames: int = 5, n_verts: int = 100) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [rng.random((n_verts, 3)).astype(np.float32) for _ in range(n_frames)]


# --- GroundPlaneAligner tests ---

def test_ground_plane_aligner_returns_4x4():
    from backend.app.export.ground_plane import GroundPlaneAligner
    aligner = GroundPlaneAligner()
    seq = _make_joint_sequence()
    T = aligner.align(seq)
    assert T.shape == (4, 4), f"Expected (4, 4), got {T.shape}"
    assert T.dtype == np.float32


def test_ground_plane_aligner_apply_transform():
    from backend.app.export.ground_plane import GroundPlaneAligner
    aligner = GroundPlaneAligner()
    seq = _make_joint_sequence(n_frames=5)
    T = aligner.align(seq)
    result = aligner.apply_transform(seq, T)
    assert len(result) == 5
    assert result[0].shape == (70, 3)


def test_ground_plane_aligner_empty_sequence():
    from backend.app.export.ground_plane import GroundPlaneAligner
    aligner = GroundPlaneAligner()
    T = aligner.align([])
    assert T.shape == (4, 4)
    np.testing.assert_array_equal(T, np.eye(4, dtype=np.float32))


# --- MLBMound tests ---

def test_mound_geometry_dtypes_and_shape():
    from backend.app.export.mound import MLBMound
    mound = MLBMound()
    mesh = mound.generate_mesh()
    assert "vertices" in mesh and "faces" in mesh
    assert mesh["vertices"].dtype == np.float32, f"vertices dtype: {mesh['vertices'].dtype}"
    assert mesh["faces"].dtype == np.int32, f"faces dtype: {mesh['faces'].dtype}"
    assert mesh["vertices"].ndim == 2 and mesh["vertices"].shape[1] == 3
    assert mesh["faces"].ndim == 2 and mesh["faces"].shape[1] == 3
    # At minimum: 1 center + rings * segments vertices
    assert mesh["vertices"].shape[0] > 100


def test_mound_plate_direction():
    from backend.app.export.mound import MLBMound
    mound = MLBMound()
    d = mound.generate_plate_direction()
    assert d.shape == (3,)
    np.testing.assert_allclose(np.linalg.norm(d), 1.0, atol=1e-6)


# --- GLBExporter tests ---

@requires_pygltflib
def test_glb_exporter_creates_file(tmp_path):
    from backend.app.export.glb import GLBExporter
    exporter = GLBExporter(vertex_count=100, face_count=50)
    frames = _make_mesh_frames(n_frames=5, n_verts=100)
    meta = {
        "pitch_type": "FF",
        "pitcher_id": "test-001",
        "frame_timestamps": list(range(5)),
        "phase_markers": {"stride": 1, "release": 3},
    }
    out = str(tmp_path / "test_pitch.glb")
    result = exporter.export_pitch(frames, meta, out)
    assert result == out
    assert os.path.exists(out)
    assert out.endswith(".glb")
    assert os.path.getsize(out) > 0


@requires_pygltflib
def test_glb_exporter_export_sequence_alias(tmp_path):
    from backend.app.export.glb import GLBExporter
    exporter = GLBExporter()
    frames = _make_mesh_frames(n_frames=3, n_verts=50)
    out = str(tmp_path / "seq.glb")
    result = exporter.export_sequence(frames, {}, out)
    assert os.path.exists(result)


# --- ComparisonGLBBuilder tests ---

@requires_pygltflib
def test_comparison_builder(tmp_path):
    from backend.app.export.comparison import ComparisonGLBBuilder
    builder = ComparisonGLBBuilder()
    frames1 = _make_mesh_frames(n_frames=5, n_verts=80)
    frames2 = _make_mesh_frames(n_frames=7, n_verts=80)
    meta1 = {"pitch_type": "FF", "pitcher_id": "p1"}
    meta2 = {"pitch_type": "SL", "pitcher_id": "p2"}
    out = str(tmp_path / "comparison.glb")
    result = builder.build(frames1, frames2, meta1, meta2, out)
    assert os.path.exists(result)
    assert result.endswith(".glb")


def test_comparison_phase_align():
    from backend.app.export.comparison import ComparisonGLBBuilder
    builder = ComparisonGLBBuilder()
    frames1 = _make_mesh_frames(n_frames=5, n_verts=50)
    frames2 = _make_mesh_frames(n_frames=10, n_verts=50)
    a1, a2 = builder.phase_align(frames1, frames2)
    assert len(a1) == len(a2) == 10
