---
name: mlx-weight-converter
phase: 1
sprint: 1
parent: mlx-port-manifest
depends_on: []
status: draft
created: 2026-04-03
---

# MLX Weight Converter

Convert SAM 3D Body PyTorch checkpoint + JIT MHR model to MLX safetensors format.

## Requirements

- Convert `model.ckpt` (2.1GB PyTorch state_dict) to safetensors
- Convert `mhr_model.pt` (696MB JIT TorchScript) to safetensors
- Split all fused QKV weights into separate Q, K, V tensors
- Transpose Conv2d weights from PyTorch NCHW to MLX NHWC
- Handle ConvTranspose2d weight transposition
- Map PyTorch key names to MLX naming conventions
- Preserve all parameter dtypes (float32, float16 where applicable)

## Acceptance Criteria

- [ ] Running converter produces a directory with `.safetensors` files + `config.json`
- [ ] Total safetensors size within 5% of original checkpoint size
- [ ] All weight shapes validated against expected MLX module shapes
- [ ] Round-trip test: load in MLX, verify shapes and dtypes match

## Technical Approach

Follow the SAM 3.1 converter pattern (`/tmp/mlx-vlm/mlx_vlm/models/sam3_1/convert_weights.py`):

1. Load PyTorch checkpoint with `torch.load(..., map_location="cpu")`
2. Build key mapping table: PyTorch prefix → MLX prefix
3. QKV splitting: detect `qkv.weight`/`in_proj_weight` patterns, split `(3*D, D)` → 3 × `(D, D)`
4. Conv transpose: `(out, in, kH, kW)` → `(out, kH, kW, in)` for Conv2d
5. ConvTranspose: `(in, out, kH, kW)` → `(out, kH, kW, in)`
6. Save via `safetensors.numpy.save_file()`
7. Generate `config.json` with model architecture params

### Key Mapping (DINOv3 → MLX)

```
backbone.encoder.patch_embed.*     → embeddings.patch_embeddings.*
backbone.encoder.blocks.{i}.*     → layers.{i}.*
backbone.encoder.blocks.{i}.attn.qkv.weight → layers.{i}.attention.{q,k,v}_proj.weight
backbone.encoder.blocks.{i}.mlp.* → layers.{i}.mlp.*
backbone.encoder.norm.*           → layer_norm.*
head_pose.proj.*                  → head_pose.proj.*
head_pose.mhr.*                   → head_pose.mhr.* (MHR needs special handling)
```

### MHR JIT Extraction

The JIT model stores the character skeleton, blend shapes, and skinning weights as buffers.
Extract via `torch.jit.load()` then iterate `named_parameters()` and `named_buffers()`.

## Files

| File | Action |
|------|--------|
| `sam3d_mlx/convert_weights.py` | create |
| `sam3d_mlx/config.py` | create (model config dataclass) |

## Tasks

1. Create key mapping table from PyTorch state_dict keys
2. Implement QKV splitting + conv transposition logic
3. Extract JIT MHR parameters and buffers
4. Write safetensors output + config.json generator
5. Validate output shapes against expected MLX module dims
