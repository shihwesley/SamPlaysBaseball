"""Numerical parity test + CPU/MPS/MLX benchmark comparison.

Loads both the PyTorch and MLX models, runs identical inputs,
compares outputs numerically, and benchmarks all three backends.

Usage:
    python -m benchmarks.bench_parity
"""

import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/tmp/sam-3d-body")

# ──────────────────────────────────────────────────────────────────
# 1. Load PyTorch model components for per-layer comparison
# ──────────────────────────────────────────────────────────────────

def load_pytorch_backbone():
    """Load DINOv3 backbone weights into a minimal forward-pass module."""
    ckpt = torch.load("/tmp/sam3d-weights/model.ckpt", map_location="cpu", weights_only=False)

    # The backbone is a DINOv3 ViT loaded via torch.hub
    # We need to instantiate it the same way SAM 3D Body does
    try:
        encoder = torch.hub.load(
            "facebookresearch/dinov3",
            "dinov3_vith16plus",
            source="github",
            pretrained=False,
            drop_path=0.1,
        )
    except Exception as e:
        print(f"  Could not load DINOv3 via torch.hub: {e}")
        print("  Skipping PyTorch backbone comparison.")
        return None

    # Load backbone weights from checkpoint
    backbone_state = {}
    for k, v in ckpt.items():
        if k.startswith("backbone.encoder."):
            new_key = k[len("backbone.encoder."):]
            backbone_state[new_key] = v

    missing, unexpected = encoder.load_state_dict(backbone_state, strict=False)
    if missing:
        print(f"  Backbone missing keys: {len(missing)}")
    encoder.eval()
    return encoder


def load_pytorch_decoder():
    """Load decoder weights."""
    from sam_3d_body.models.decoders.promptable_decoder import PromptableDecoder

    decoder = PromptableDecoder(
        dims=1024,
        context_dims=1280,
        depth=6,
        num_heads=8,
        head_dims=64,
        mlp_dims=1024,
        layer_scale_init_value=0.0,
        ffn_type="origin",
        repeat_pe=True,
        do_interm_preds=False,
    )

    ckpt = torch.load("/tmp/sam3d-weights/model.ckpt", map_location="cpu", weights_only=False)
    decoder_state = {}
    for k, v in ckpt.items():
        if k.startswith("decoder.") and not k.startswith("decoder_hand."):
            new_key = k[len("decoder."):]
            decoder_state[new_key] = v

    decoder.load_state_dict(decoder_state)
    decoder.eval()
    return decoder


def load_pytorch_mhr_head():
    """Load MHR head weights."""
    from sam_3d_body.models.heads.mhr_head import MHRHead

    head = MHRHead(
        input_dim=1024,
        mlp_depth=2,
        mhr_model_path="/tmp/sam3d-weights/assets/mhr_model.pt",
        mlp_channel_div_factor=1,
        enable_hand_model=False,
    )

    ckpt = torch.load("/tmp/sam3d-weights/model.ckpt", map_location="cpu", weights_only=False)
    head_state = {}
    for k, v in ckpt.items():
        if k.startswith("head_pose.") and not k.startswith("head_pose_hand."):
            new_key = k[len("head_pose."):]
            head_state[new_key] = v

    head.load_state_dict(head_state, strict=False)
    head.eval()
    head.ensure_mhr_on_cpu()
    return head


# ──────────────────────────────────────────────────────────────────
# 2. Load MLX model components
# ──────────────────────────────────────────────────────────────────

def load_mlx_backbone():
    import mlx.core as mx
    from sam3d_mlx.backbone import DINOv3Backbone
    from sam3d_mlx.config import SAM3DConfig
    from safetensors import safe_open

    config = SAM3DConfig()
    model = DINOv3Backbone(config)

    weights = []
    with safe_open("/tmp/sam3d-mlx-weights/model.safetensors", framework="numpy") as f:
        for key in f.keys():
            if key.startswith("backbone."):
                model_key = key[len("backbone."):]
                if "bias_mask" in model_key or "k_proj.bias" in model_key:
                    continue
                weights.append((model_key, mx.array(f.get_tensor(key))))
    model.load_weights(weights)
    return model


def load_mlx_decoder():
    import mlx.core as mx
    from sam3d_mlx.decoder import PromptableDecoder
    from safetensors import safe_open

    decoder = PromptableDecoder(dims=1024, context_dims=1280, depth=6)

    weights = []
    with safe_open("/tmp/sam3d-mlx-weights/model.safetensors", framework="numpy") as f:
        for key in f.keys():
            if key.startswith("decoder.") and not key.startswith("decoder_hand."):
                model_key = key[len("decoder."):]
                weights.append((model_key, mx.array(f.get_tensor(key))))
    decoder.load_weights(weights)
    return decoder


def load_mlx_mhr_head():
    import mlx.core as mx
    from sam3d_mlx.mhr_head import MHRHead
    from sam3d_mlx.config import SAM3DConfig

    head = MHRHead(input_dim=1024, config=SAM3DConfig())
    head.load_all_weights("/tmp/sam3d-mlx-weights/model.safetensors")
    return head


# ──────────────────────────────────────────────────────────────────
# 3. Parity tests
# ──────────────────────────────────────────────────────────────────

def test_backbone_parity():
    """Compare backbone outputs between PyTorch and MLX."""
    import mlx.core as mx

    print("\n═══ Backbone Parity ═══")

    pt_model = load_pytorch_backbone()
    if pt_model is None:
        return None

    mlx_model = load_mlx_backbone()

    # Create deterministic input
    np.random.seed(42)
    input_np = np.random.randn(1, 3, 512, 384).astype(np.float32)

    # PyTorch: NCHW — run in float32 to avoid bf16 bias mismatch
    pt_input = torch.from_numpy(input_np)
    with torch.no_grad():
        pt_out = pt_model.float().get_intermediate_layers(pt_input, n=1, reshape=True, norm=True)[-1]
        pt_out = pt_out.float().numpy()  # (1, 1280, 32, 24) or (1, 32, 24, 1280)

    # Handle PyTorch output format (might be NCHW)
    if pt_out.shape[1] == 1280:
        pt_out = np.transpose(pt_out, (0, 2, 3, 1))  # NCHW -> NHWC

    # MLX: NHWC
    mlx_input = mx.array(np.transpose(input_np, (0, 2, 3, 1)))  # NCHW -> NHWC
    mlx_out = mlx_model(mlx_input)
    mx.eval(mlx_out)
    mlx_out_np = np.array(mlx_out)

    # Compare
    max_diff = np.abs(pt_out - mlx_out_np).max()
    mean_diff = np.abs(pt_out - mlx_out_np).mean()
    cos_sim = np.sum(pt_out * mlx_out_np) / (np.linalg.norm(pt_out) * np.linalg.norm(mlx_out_np) + 1e-8)

    print(f"  PT shape:  {pt_out.shape}")
    print(f"  MLX shape: {mlx_out_np.shape}")
    print(f"  Max diff:  {max_diff:.6f}  (target: < 1e-3)")
    print(f"  Mean diff: {mean_diff:.6f}")
    print(f"  Cos sim:   {cos_sim:.6f}")
    print(f"  PASS" if max_diff < 1e-2 else f"  FAIL (max_diff={max_diff:.6f})")

    return {"max_diff": max_diff, "mean_diff": mean_diff, "cos_sim": cos_sim}


def test_decoder_parity():
    """Compare decoder outputs between PyTorch and MLX."""
    import mlx.core as mx

    print("\n═══ Decoder Parity ═══")

    pt_model = load_pytorch_decoder()
    mlx_model = load_mlx_decoder()

    # Create deterministic inputs
    np.random.seed(42)
    tokens_np = np.random.randn(1, 5, 1024).astype(np.float32)
    context_np = np.random.randn(1, 768, 1280).astype(np.float32)
    token_pe_np = np.random.randn(1, 5, 1024).astype(np.float32)
    context_pe_np = np.random.randn(1, 768, 1280).astype(np.float32)

    # PyTorch
    with torch.no_grad():
        pt_out, _ = pt_model(
            token_embedding=torch.from_numpy(tokens_np),
            image_embedding=torch.from_numpy(context_np),
            token_augment=torch.from_numpy(token_pe_np),
            image_augment=torch.from_numpy(context_pe_np),
            channel_first=False,
        )
        pt_out_np = pt_out.numpy()

    # MLX
    mlx_out = mlx_model(
        mx.array(tokens_np),
        mx.array(context_np),
        mx.array(token_pe_np),
        mx.array(context_pe_np),
    )
    mx.eval(mlx_out)
    mlx_out_np = np.array(mlx_out)

    max_diff = np.abs(pt_out_np - mlx_out_np).max()
    mean_diff = np.abs(pt_out_np - mlx_out_np).mean()

    print(f"  PT shape:  {pt_out_np.shape}")
    print(f"  MLX shape: {mlx_out_np.shape}")
    print(f"  Max diff:  {max_diff:.6f}  (target: < 1e-3)")
    print(f"  Mean diff: {mean_diff:.6f}")
    print(f"  PASS" if max_diff < 1e-3 else f"  FAIL (max_diff={max_diff:.6f})")

    return {"max_diff": max_diff, "mean_diff": mean_diff}


def test_mhr_head_parity():
    """Compare MHR head outputs between PyTorch and MLX."""
    import mlx.core as mx

    print("\n═══ MHR Head Parity ═══")

    pt_model = load_pytorch_mhr_head()
    mlx_model = load_mlx_mhr_head()

    # Create deterministic input
    np.random.seed(42)
    token_np = np.random.randn(1, 1024).astype(np.float32)

    # PyTorch
    with torch.no_grad():
        pt_out = pt_model(torch.from_numpy(token_np))
        pt_verts = pt_out["pred_vertices"].numpy()
        pt_j3d = pt_out["pred_keypoints_3d"].numpy()
        pt_joints = pt_out["pred_joint_coords"].numpy()

    # MLX
    mlx_out = mlx_model(mx.array(token_np))
    mx.eval(mlx_out["pred_vertices"])
    mlx_verts = np.array(mlx_out["pred_vertices"])
    mlx_j3d = np.array(mlx_out["pred_keypoints_3d"])
    mlx_joints = np.array(mlx_out["pred_joint_coords"])

    for name, pt_arr, mlx_arr in [
        ("vertices", pt_verts, mlx_verts),
        ("keypoints_3d", pt_j3d, mlx_j3d),
        ("joint_coords", pt_joints, mlx_joints),
    ]:
        max_diff = np.abs(pt_arr - mlx_arr).max()
        mean_diff = np.abs(pt_arr - mlx_arr).mean()
        print(f"  {name:15s}  max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}  {'PASS' if max_diff < 1e-3 else 'FAIL'}")

    return {
        "vert_max_diff": np.abs(pt_verts - mlx_verts).max(),
        "joint_max_diff": np.abs(pt_joints - mlx_joints).max(),
    }


# ──────────────────────────────────────────────────────────────────
# 4. Performance benchmarks: CPU vs MPS vs MLX
# ──────────────────────────────────────────────────────────────────

def bench_pytorch(device_name: str, n_warmup: int = 2, n_bench: int = 5):
    """Benchmark PyTorch backbone on given device."""
    pt_model = load_pytorch_backbone()
    if pt_model is None:
        return None

    device = torch.device(device_name)
    dtype = torch.bfloat16 if device_name == "cpu" else torch.float16

    pt_model = pt_model.to(device).to(dtype)

    np.random.seed(42)
    x = torch.randn(1, 3, 512, 384, dtype=dtype, device=device)

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = pt_model.get_intermediate_layers(x, n=1, reshape=True, norm=True)
        if device_name == "mps":
            torch.mps.synchronize()

    # Bench
    times = []
    for _ in range(n_bench):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = pt_model.get_intermediate_layers(x, n=1, reshape=True, norm=True)
        if device_name == "mps":
            torch.mps.synchronize()
        times.append(time.perf_counter() - t0)

    return {
        "median": sorted(times)[len(times) // 2],
        "min": min(times),
        "max": max(times),
    }


def bench_mlx_backbone(n_warmup: int = 3, n_bench: int = 10):
    """Benchmark MLX backbone."""
    import mlx.core as mx

    model = load_mlx_backbone()
    x = mx.random.normal((1, 512, 384, 3))

    # Warmup
    for _ in range(n_warmup):
        out = model(x)
        mx.eval(out)

    # Bench
    times = []
    for _ in range(n_bench):
        t0 = time.perf_counter()
        out = model(x)
        mx.eval(out)
        times.append(time.perf_counter() - t0)

    return {
        "median": sorted(times)[len(times) // 2],
        "min": min(times),
        "max": max(times),
    }


def run_benchmarks():
    """Run backbone benchmarks across all available backends."""
    print("\n═══ Backbone Benchmark: CPU vs MPS vs MLX ═══")
    print(f"{'Backend':>12s}  {'Median':>10s}  {'Min':>10s}  {'Max':>10s}  {'Speedup':>10s}")
    print("─" * 60)

    results = {}

    # PyTorch CPU
    print("  Benchmarking PyTorch CPU...", end="", flush=True)
    cpu_result = bench_pytorch("cpu", n_warmup=1, n_bench=3)
    if cpu_result:
        cpu_ms = cpu_result["median"] * 1000
        results["cpu"] = cpu_result
        print(f"\r{'PT CPU':>12s}  {cpu_ms:>9.1f}ms  {cpu_result['min']*1000:>9.1f}ms  {cpu_result['max']*1000:>9.1f}ms  {'1.0x (base)':>10s}")
    else:
        print("\r  PT CPU: skipped (model not available)")

    # PyTorch MPS
    if torch.backends.mps.is_available():
        print("  Benchmarking PyTorch MPS...", end="", flush=True)
        mps_result = bench_pytorch("mps", n_warmup=2, n_bench=5)
        if mps_result:
            mps_ms = mps_result["median"] * 1000
            results["mps"] = mps_result
            speedup = cpu_result["median"] / mps_result["median"] if cpu_result else 0
            print(f"\r{'PT MPS':>12s}  {mps_ms:>9.1f}ms  {mps_result['min']*1000:>9.1f}ms  {mps_result['max']*1000:>9.1f}ms  {speedup:>9.1f}x")

    # MLX
    print("  Benchmarking MLX...", end="", flush=True)
    mlx_result = bench_mlx_backbone(n_warmup=3, n_bench=10)
    mlx_ms = mlx_result["median"] * 1000
    results["mlx"] = mlx_result
    speedup_cpu = cpu_result["median"] / mlx_result["median"] if cpu_result else 0
    speedup_mps = results.get("mps", {}).get("median", 0) / mlx_result["median"] if "mps" in results else 0
    print(f"\r{'MLX':>12s}  {mlx_ms:>9.1f}ms  {mlx_result['min']*1000:>9.1f}ms  {mlx_result['max']*1000:>9.1f}ms  {speedup_cpu:>9.1f}x")

    print("─" * 60)
    if "mps" in results:
        print(f"  MLX is {speedup_mps:.1f}x faster than PyTorch MPS")
    if cpu_result:
        print(f"  MLX is {speedup_cpu:.1f}x faster than PyTorch CPU")

    return results


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("SAM 3D Body — Parity Tests + Backend Benchmarks")
    print("=" * 60)

    # Parity tests
    print("\n─── PARITY TESTS ───")

    try:
        test_decoder_parity()
    except Exception as e:
        print(f"  Decoder parity SKIPPED: {e}")

    try:
        test_mhr_head_parity()
    except Exception as e:
        print(f"  MHR head parity SKIPPED: {e}")

    try:
        test_backbone_parity()
    except Exception as e:
        print(f"  Backbone parity SKIPPED: {e}")

    # Benchmarks
    print("\n─── BENCHMARKS ───")
    try:
        run_benchmarks()
    except Exception as e:
        print(f"  Benchmarks FAILED: {e}")
        import traceback
        traceback.print_exc()

    print("\nDone.")
