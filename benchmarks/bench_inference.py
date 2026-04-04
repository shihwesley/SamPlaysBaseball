"""Performance benchmarks for SAM 3D Body MLX inference.

Run: python -m benchmarks.bench_inference

Reports per-component and full-pipeline latency with median/p95/p99 stats.
"""

import os
import sys
import time

import mlx.core as mx
import numpy as np

from sam3d_mlx.config import SAM3DConfig
from sam3d_mlx.estimator import SAM3DBodyEstimator
from sam3d_mlx.model import SAM3DBody

WEIGHTS_PATH = "/tmp/sam3d-mlx-weights/"
WARMUP_ITERS = 3
BENCH_ITERS = 10


def _percentile(times, p):
    s = sorted(times)
    idx = int(len(s) * p / 100)
    idx = min(idx, len(s) - 1)
    return s[idx]


def _stats(times):
    s = sorted(times)
    return {
        "median": s[len(s) // 2],
        "p95": _percentile(times, 95),
        "p99": _percentile(times, 99),
        "min": s[0],
        "max": s[-1],
    }


def bench_backbone(model, iters=BENCH_ITERS):
    """Benchmark backbone (DINOv3 ViT-H+) forward pass."""
    x = mx.zeros((1, 512, 384, 3))

    # Warmup
    for _ in range(WARMUP_ITERS):
        out = model.backbone(x)
        mx.eval(out)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = model.backbone(x)
        mx.eval(out)
        times.append(time.perf_counter() - t0)

    return _stats(times)


def bench_decoder(model, iters=BENCH_ITERS):
    """Benchmark decoder forward pass with pre-built tokens."""
    B = 1
    config = model.config
    tokens = mx.zeros((B, 75, config.decoder_dim))
    image_features = mx.zeros((B, 32, 24, config.embed_dim))
    token_pe = mx.zeros((B, 75, config.decoder_dim))
    image_pe = mx.zeros((1, 32, 24, config.prompt_embed_dim))

    # Warmup
    for _ in range(WARMUP_ITERS):
        out = model.decoder(tokens, image_features, token_pe, image_pe)
        mx.eval(out)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = model.decoder(tokens, image_features, token_pe, image_pe)
        mx.eval(out)
        times.append(time.perf_counter() - t0)

    return _stats(times)


def bench_mhr_head(model, iters=BENCH_ITERS):
    """Benchmark MHR head: pose token -> body mesh."""
    pose_token = mx.zeros((1, 1024))
    init_est = mx.zeros((1, 519))

    # Warmup
    for _ in range(WARMUP_ITERS):
        out = model.head_pose(pose_token, init_estimate=init_est)
        mx.eval(out["pred_vertices"])

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = model.head_pose(pose_token, init_estimate=init_est)
        mx.eval(out["pred_vertices"])
        mx.eval(out["pred_keypoints_3d"])
        mx.eval(out["pred_joint_coords"])
        times.append(time.perf_counter() - t0)

    return _stats(times)


def bench_camera_head(model, iters=BENCH_ITERS):
    """Benchmark camera head."""
    pose_token = mx.zeros((1, 1024))
    init_cam = mx.zeros((1, 3))

    for _ in range(WARMUP_ITERS):
        out = model.head_camera(pose_token, init_estimate=init_cam)
        mx.eval(out)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = model.head_camera(pose_token, init_estimate=init_cam)
        mx.eval(out)
        times.append(time.perf_counter() - t0)

    return _stats(times)


def bench_full_pipeline(model, iters=BENCH_ITERS):
    """Benchmark full model forward pass (image -> mesh + camera)."""
    x = mx.zeros((1, 512, 384, 3))
    cliff = mx.zeros((1, 3))

    # Warmup
    for _ in range(WARMUP_ITERS):
        body_out, cam = model(x, cliff_condition=cliff)
        mx.eval(body_out["pred_vertices"])
        mx.eval(cam)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        body_out, cam = model(x, cliff_condition=cliff)
        mx.eval(body_out["pred_vertices"])
        mx.eval(body_out["pred_keypoints_3d"])
        mx.eval(body_out["pred_joint_coords"])
        mx.eval(cam)
        times.append(time.perf_counter() - t0)

    return _stats(times)


def bench_estimator_predict(iters=BENCH_ITERS):
    """Benchmark the high-level estimator.predict() including preprocessing."""
    est = SAM3DBodyEstimator(WEIGHTS_PATH)
    img = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    bbox = [100, 50, 400, 550]

    # Warmup
    for _ in range(WARMUP_ITERS):
        est.predict(img, bbox=bbox)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        est.predict(img, bbox=bbox)
        times.append(time.perf_counter() - t0)

    return _stats(times)


def bench_preprocessing(iters=50):
    """Benchmark image preprocessing (crop, resize, normalize)."""
    from sam3d_mlx.batch_prep import prepare_image, get_cliff_condition

    img = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    bbox = [100, 50, 400, 550]

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        prepare_image(img, bbox)
        get_cliff_condition(bbox, (640, 480))
        times.append(time.perf_counter() - t0)

    return _stats(times)


def estimate_memory():
    """Estimate model memory footprint from weight sizes."""
    from pathlib import Path
    weights_dir = Path(WEIGHTS_PATH)
    total_bytes = 0
    for f in weights_dir.glob("*.safetensors"):
        total_bytes += f.stat().st_size
    return total_bytes


def main():
    print("=" * 70)
    print("SAM 3D Body MLX -- Performance Benchmarks")
    print("=" * 70)
    print(f"Weights:     {WEIGHTS_PATH}")
    print(f"Warmup:      {WARMUP_ITERS} iters")
    print(f"Bench:       {BENCH_ITERS} iters")
    print()

    # Load model
    print("Loading model...", end="", flush=True)
    t0 = time.perf_counter()
    config = SAM3DConfig()
    model = SAM3DBody(config)
    model.load_all_weights(WEIGHTS_PATH)
    model.eval()
    load_time = time.perf_counter() - t0
    print(f" {load_time:.1f}s")

    # Memory estimate
    mem_bytes = estimate_memory()
    print(f"Weight size: {mem_bytes / 1e6:.1f} MB (on-disk safetensors)")
    print()

    # Run benchmarks
    results = {}

    components = [
        ("Preprocessing", lambda: bench_preprocessing()),
        ("Backbone (ViT-H+)", lambda: bench_backbone(model)),
        ("Decoder (6-layer)", lambda: bench_decoder(model)),
        ("MHR Head (body model)", lambda: bench_mhr_head(model)),
        ("Camera Head", lambda: bench_camera_head(model)),
        ("Full Pipeline (model only)", lambda: bench_full_pipeline(model)),
        ("Estimator.predict (e2e)", lambda: bench_estimator_predict()),
    ]

    for name, bench_fn in components:
        print(f"Benchmarking {name}...", end="", flush=True)
        try:
            stats = bench_fn()
            results[name] = stats
            print(f" {stats['median']*1000:.1f}ms")
        except Exception as e:
            results[name] = None
            print(f" ERROR: {e}")

    # Print results table
    print()
    print("-" * 70)
    print(f"{'Component':<30} {'Median':>8} {'P95':>8} {'P99':>8} {'Min':>8} {'Max':>8}")
    print("-" * 70)
    for name, stats in results.items():
        if stats is None:
            print(f"{name:<30} {'ERROR':>8}")
            continue
        print(
            f"{name:<30} "
            f"{stats['median']*1000:>7.1f}ms "
            f"{stats['p95']*1000:>7.1f}ms "
            f"{stats['p99']*1000:>7.1f}ms "
            f"{stats['min']*1000:>7.1f}ms "
            f"{stats['max']*1000:>7.1f}ms"
        )
    print("-" * 70)

    # Target check
    full = results.get("Full Pipeline (model only)")
    if full:
        median_ms = full["median"] * 1000
        target = 500
        status = "PASS" if median_ms < target else "MISS"
        print(f"\nTarget: body-mode < {target}ms --> {status} ({median_ms:.0f}ms)")

    e2e = results.get("Estimator.predict (e2e)")
    if e2e:
        print(f"Throughput: {1.0 / e2e['median']:.1f} images/sec (batch=1)")

    print()


if __name__ == "__main__":
    main()
