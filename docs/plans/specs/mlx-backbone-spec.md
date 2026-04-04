---
name: mlx-backbone
phase: 1
sprint: 2
parent: mlx-port-manifest
depends_on: [mlx-weight-converter]
status: draft
created: 2026-04-03
---

# MLX Backbone — DINOv3-H+ Vision Transformer

Port the DINOv3-H+ ViT backbone (840M params) to MLX.

## Requirements

- Pure MLX implementation of DinoVisionTransformer
- RoPE position encoding (2D axial, with configurable base/normalization)
- SwiGLU FFN (gate-up-down pattern with aligned dimensions)
- LayerNorm with bf16 eps=1e-5 (NOT RMSNorm — checkpoint uses layernormbf16)
- LayerScale (learnable per-block scaling)
- SelfAttentionBlock with `mx.fast.scaled_dot_product_attention`
- PatchEmbed via Conv2d (patch_size=16, embed_dim=1280 for H+ variant)
- CLS token + optional storage tokens
- `get_intermediate_layers()` API for SAM 3D Body's single-layer extraction
- Channel-last (NHWC) throughout

## Architecture (DINOv3-H+ specific)

| Parameter | Value |
|-----------|-------|
| embed_dim | 1280 |
| depth | 32 |
| num_heads | 20 |
| head_dim | 64 (1280/20) |
| ffn_layer | swiglu (no alignment) |
| ffn_ratio | 6.0 |
| patch_size | 16 |
| norm | LayerNorm (bf16 eps=1e-5, NOT RMSNorm) |
| pos_embed | RoPE (2D axial, base=100, rescale_coords=2, dtype=fp32) |
| input_size | 512×384 (SAM 3D Body default) |
| storage_tokens | 4 |
| mask_k_bias | True |
| qkv_bias | True |
| proj_bias | True |
| ffn_bias | True |

## Acceptance Criteria

- [ ] Forward pass produces same shape output as PyTorch: `(B, H/16, W/16, 1280)`
- [ ] Numerical parity: max per-element diff < 1e-3 vs PyTorch on same input
- [ ] Loads converted safetensors weights without shape mismatches
- [ ] Memory usage within 10% of PyTorch CPU baseline
- [ ] Single-image inference < 500ms on M-series (vs ~1.5s MPS PyTorch)

## Technical Approach

Follow SAM 3 MLX ViT pattern (`/tmp/mlx-vlm/mlx_vlm/models/sam3/vision.py`).

### Key Differences from SAM 3 ViT

| SAM 3 ViT | DINOv3-H+ ViT |
|-----------|----------------|
| LayerNorm (eps=1e-6) | LayerNorm (bf16 eps=1e-5) |
| GELU MLP | SwiGLU FFN (w1/w2/w3) |
| No LayerScale | LayerScale (init=1e-5) |
| Windowed + global attn | All global (no windows) |
| Fused QKV | Fused QKV + bias_mask (mask_k_bias=True) |
| Absolute pos embed + RoPE | RoPE only (no absolute, rescale_coords=2) |
| 16 heads | 20 heads |
| ffn_ratio=4.0 | ffn_ratio=6.0 |
| No storage tokens | 4 storage tokens |

### MLX Module Structure

```
DINOv3Backbone(nn.Module)
├── patch_embed: PatchEmbed (Conv2d)
├── cls_token: mx.array (1, 1, 1280)
├── storage_tokens: mx.array (1, 4, 1280)
├── rope_embed: RoPE2D (base=100, rescale_coords=2, dtype=fp32)
├── blocks: [SelfAttentionBlock × 32]
│   ├── norm1: LayerNorm (eps=1e-5)
│   ├── attention: Attention (fused qkv + bias_mask + RoPE, 20 heads)
│   ├── ls1: LayerScale (init=1e-5)
│   ├── norm2: LayerNorm (eps=1e-5)
│   ├── mlp: SwiGLU (w1, w2, w3, ratio=6.0)
│   └── ls2: LayerScale (init=1e-5)
└── norm: LayerNorm (eps=1e-5)
```

## Files

| File | Action |
|------|--------|
| `sam3d_mlx/backbone.py` | create (DINOv3 ViT) |
| `sam3d_mlx/rope.py` | create (2D axial RoPE) |
| `sam3d_mlx/layers.py` | create (shared: LayerNorm32, SwiGLU, LayerScale, DropPath) |
| `tests/test_backbone.py` | create (parity test) |

## Tasks

1. Implement shared layers.py: LayerNorm32, SwiGLU (w1/w2/w3 gate*silu(x) pattern, ratio=6.0), LayerScale (init=1e-5), DropPath
2. Implement RoPE 2D axial position encoding (base=100, rescale_coords=2, dtype=fp32)
3. Implement SelfAttentionBlock with LayerNorm + LayerScale + mx.fast.sdpa + bias_mask (20 heads, head_dim=64)
4. Implement PatchEmbed + CLS token + storage_tokens (4) + get_intermediate_layers
5. Write numerical parity test comparing MLX vs PyTorch backbone output
