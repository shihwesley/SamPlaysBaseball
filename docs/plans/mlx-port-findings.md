# MLX Port — Findings & Decisions

## Goal
Port SAM 3D Body from PyTorch to MLX with TurboQuant, numerical parity, no PyTorch at inference.

## Priority
Quality

## Mode
Spec-driven (sub-plan under sam3d-inference)

## Approach
Follow the SAM 3.1 mlx-vlm porting pattern. Convert weights to safetensors, rewrite each component (backbone, decoder, MHR) in MLX using `mlx.nn` and `mx.fast.scaled_dot_product_attention`. Integrate TurboQuant KV cache from mlx-vlm for decoder cross-attention.

## Requirements (validated)
- Numerical parity with PyTorch (<1e-3 per element)
- Weight converter: .ckpt + JIT .pt → safetensors
- Full video pipeline on MLX, no PyTorch at inference
- TurboQuant for decoder attention KV cache

## Spec Map
→ Manifest: specs/mlx-port-manifest.md
→ Specs directory: docs/plans/specs/mlx-*.md

### Per-Spec Decisions
| Spec | Key Decision | Rationale | Affects |
|------|-------------|-----------|---------|
| backbone | Use SAM 3 ViT pattern, not custom | Proven pattern, minimal adaptation needed | decoder (input format) |
| decoder | TurboQuant at 3.5 bits default | Best quality/compression balance per mlx-vlm benchmarks | inference memory |
| mhr-head | Pure MLX FK, not JIT wrapper | Eliminates PyTorch dependency entirely | inference (no torch) |

## Research Context
| Topic | Expertise | Knowledge Store |
|-------|-----------|----------------|
| sam3d-baseball | ~/.claude/research/sam3d-baseball/expertise.md | ~/.neo-research/knowledge/sam3d-baseball.mv2 |

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Channel-last (NHWC) throughout | MLX convention, avoids transposes at boundaries |
| Precomputed RoPE tables | Avoids runtime compute, matches SAM 3 pattern |
| Sequential FK loop | Skeleton hierarchy is inherently sequential (127 joints) |
| safetensors format | Standard for MLX ecosystem, supports memory mapping |

## Key Findings from Code Analysis

### DINOv3-H+ Architecture (from source)
- 32 blocks, embed_dim=1280, 16 heads, head_dim=80
- SwiGLU FFN with 64-alignment (not GELU MLP)
- RMSNorm (not LayerNorm)
- LayerScale (learnable per-block scalar)
- RoPE with base=100, separate normalization per axis
- CLS token + mask_token (mask_token used for training only)

### SAM 3.1 MLX Port Pattern (from mlx-vlm)
- convert_weights.py: QKV split + conv transpose + key remapping
- vision.py: ViTBackbone with windowed/global attention + RoPE
- sam_components.py: TwoWayTransformer, OutputMLP, PromptEncoder
- All attention via mx.fast.scaled_dot_product_attention

### MHR JIT Model Structure (from mhr_head.py)
- self.mhr(shape_params, model_params, expr_params) → (vertices, skel_state)
- skel_state: (B, 127, 8) → split into coords(3), quats(4), padding(1)
- Internally: blend shapes → FK chain → LBS → pose correctives
- Uses float64 for skeleton math (blocks MPS, fine for MLX)
