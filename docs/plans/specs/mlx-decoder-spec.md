---
name: mlx-decoder
phase: 2
sprint: 1
parent: mlx-port-manifest
depends_on: [mlx-backbone]
status: draft
created: 2026-04-03
---

# MLX Decoder — Promptable Transformer Decoder + TurboQuant

Port the SAM 3D Body promptable decoder to MLX with TurboQuant KV cache.

## Requirements

- Port PromptableDecoder (cross-attention between pose tokens and image features)
- Port token_to_pose_output_fn (pose regression from decoder tokens)
- Port camera projection (perspective projection to 2D keypoints)
- Integrate TurboQuant KV cache for cross-attention layers
- Port init_to_token / prev_to_token linear projections
- Port PromptEncoder (keypoint prompt encoding)
- Port ray condition embedding

## Acceptance Criteria

- [ ] Decoder produces same shape output as PyTorch given same backbone features
- [ ] Numerical parity: max diff < 1e-3 vs PyTorch on same input
- [ ] TurboQuant at 3.5 bits reduces KV cache memory by ~4.5x
- [ ] Decoder inference < 100ms on M-series (vs ~300ms MPS PyTorch)

## Architecture

SAM 3D Body decoder is a 6-layer transformer with:
- Self-attention on pose tokens (1 token) + prompt tokens + keypoint tokens
- Cross-attention from tokens to image features (backbone output)
- Per-layer pose regression via `token_to_pose_output_fn`
- Final output: dict with pred_pose_raw, pred_cam, pred_keypoints_2d/3d

### TurboQuant Integration (Optional)

The decoder runs iterative refinement (6 layers), each producing intermediate predictions.
Standard KV cache is the default. TurboQuant is opt-in for memory savings.

```python
# Default: standard KV cache (exact parity with PyTorch)
# Optional: TurboQuant for ~4.5x KV cache memory reduction
try:
    from mlx_vlm.turboquant import TurboQuantKVCache
    cache = TurboQuantKVCache(bits=3.5)
except ImportError:
    cache = StandardKVCache()  # fallback, no mlx-vlm dependency needed
```

**Note:** MLX uses NHWC natively. Remove the PyTorch `channel_first=True` reshape
(NCHW→sequence) from PromptableDecoder.forward(). Image features arrive as (B, H, W, C)
and should be reshaped to (B, H*W, C) directly.

## Files

| File | Action |
|------|--------|
| `sam3d_mlx/decoder.py` | create (promptable decoder) |
| `sam3d_mlx/transformer.py` | create (TransformerDecoderLayer: two-way attn, LayerScale, DropPath, FFN) |
| `sam3d_mlx/prompt_encoder.py` | create (keypoint/box prompts) |
| `sam3d_mlx/camera.py` | create (perspective projection) |
| `tests/test_decoder.py` | create (parity test) |

## Tasks

1. Port TransformerDecoderLayer (two-way attention, skip-first-PE, LayerScale, DropPath, FFN) to transformer.py
2. Port PromptableDecoder (layer loop, interm predictions, channel order fix for NHWC)
3. Port PromptEncoder and ray condition embedding
4. Port token_to_pose_output_fn and camera projection
5. Implement standard KV cache with optional TurboQuant (try/except import)
6. Write decoder parity test WITHOUT TurboQuant (must match PyTorch exactly)
7. Write decoder parity test WITH TurboQuant (bounded degradation < 5e-3)
