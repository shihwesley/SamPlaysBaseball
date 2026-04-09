"""Smoke test for the PR version of SAM 3D Body in mlx-vlm.

Runs against `mlx_vlm.models.sam3d_body` instead of SamPlaysBaseball's
`sam3d_mlx`. Intended to be executed with PYTHONPATH pointing at the
mlx-vlm checkout so imports resolve to the PR branch on disk, not to
the installed pip package.

Run:
    PYTHONPATH=/Users/quartershots/Source/mlx-vlm \
        python3 tests/test_pr_smoke.py
"""

import sys
import time

import numpy as np

from mlx_vlm.models.sam3d_body.estimator import SAM3DBodyEstimator
from mlx_vlm.models.sam3d_body.batch_prep import prepare_image, get_cliff_condition

WEIGHTS_PATH = "/tmp/sam3d-mlx-weights/"


def synthetic_image(h=640, w=480, seed=42):
    rng = np.random.RandomState(seed)
    img = rng.randint(100, 200, (h, w, 3), dtype=np.uint8)
    cy, cx = h // 2, w // 2
    img[cy - 100:cy + 100, cx - 40:cx + 40] = rng.randint(30, 80, (200, 80, 3), dtype=np.uint8)
    return img


def main():
    print(f"PR smoke test — importing from: {SAM3DBodyEstimator.__module__}")

    print(f"\nLoading model from {WEIGHTS_PATH} ...")
    t0 = time.perf_counter()
    est = SAM3DBodyEstimator(WEIGHTS_PATH)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    img = synthetic_image()
    bbox = [100, 50, 400, 550]

    print("\nRunning predict ...")
    t0 = time.perf_counter()
    result = est.predict(img, bbox=bbox)
    dt = time.perf_counter() - t0
    print(f"  completed in {dt * 1000:.0f}ms")

    print(f"\nOutput keys: {sorted(result.keys())}")
    for k, v in result.items():
        arr = np.asarray(v)
        print(f"  {k:22s} shape={tuple(arr.shape)} dtype={arr.dtype}")

    checks = []
    verts = np.asarray(result["pred_vertices"])
    kp3d = np.asarray(result["pred_keypoints_3d"])
    joints = np.asarray(result["pred_joint_coords"])
    cam = np.asarray(result["pred_camera"])

    checks.append(("pred_vertices shape", verts.shape == (18439, 3)))
    checks.append(("pred_keypoints_3d shape", kp3d.shape == (70, 3)))
    checks.append(("pred_joint_coords shape", joints.shape == (127, 3)))
    checks.append(("pred_camera shape", cam.shape == (3,)))
    checks.append(("no NaN/Inf in vertices", bool(np.isfinite(verts).all())))
    checks.append(("vertices within [-3, 3] m", bool(verts.min() > -3 and verts.max() < 3)))
    checks.append(("centroid near origin", float(np.linalg.norm(verts.mean(0))) < 5.0))

    # Determinism
    r2 = est.predict(img, bbox=bbox)
    diff = float(np.max(np.abs(np.asarray(r2["pred_vertices"]) - verts)))
    checks.append((f"deterministic (max diff {diff:.2e})", diff < 1e-6))

    print("\nChecks:")
    passed = 0
    for name, ok in checks:
        mark = "PASS" if ok else "FAIL"
        print(f"  {mark}  {name}")
        if ok:
            passed += 1

    total = len(checks)
    print(f"\n{passed}/{total} checks passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
