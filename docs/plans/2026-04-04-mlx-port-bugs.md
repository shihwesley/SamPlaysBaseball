# MLX Port Bug Report — SAM 3D Body

Date: 2026-04-04
Code: `sam3d_mlx/`
Compared against: `/tmp/sam-3d-body/sam_3d_body/` (PyTorch source)

## Critical Bugs (must fix before PR)

### BUG-1: init_to_token_mhr concatenation order reversed

**File:** `sam3d_mlx/model.py:388-392`
**Severity:** CRITICAL — produces completely wrong init tokens

PyTorch (`sam3d_body/sam3d_body.py:329-334`):
```python
init_input = torch.cat([condition_info, init_estimate], dim=-1)  # condition FIRST
```

MLX (our code):
```python
init_input = mx.concatenate([init_estimate, cliff_condition], axis=1)  # estimate FIRST — WRONG
```

**Fix:** Swap the order to `mx.concatenate([cliff_condition, init_estimate], axis=1)`.

---

### BUG-2: SwiGLU fused w12 weight never split into w1/w2

**Files:** `sam3d_mlx/convert_weights.py` + `sam3d_mlx/layers.py:7-17`
**Severity:** CRITICAL — backbone FFN layers have uninitialized weights

PyTorch uses a fused gate+up projection:
```python
# In SwiGLUFFNFused
self.w12 = nn.Linear(embed_dims, 2 * hidden_dims)  # single fused weight
# Forward: x1, x2 = self.w12(x).chunk(2, dim=-1)
```

MLX port uses separate projections:
```python
self.w1 = nn.Linear(embed_dim, hidden_dim)  # gate
self.w2 = nn.Linear(embed_dim, hidden_dim)  # up
```

But `convert_weights.py` never splits the `w12` weight into `w1` and `w2`. The key mapping just passes through `mlp.w12.weight` → no match in MLX model. With `strict=False`, these weights stay at initialization values.

**Fix:** In `convert_weights.py`, when encountering a key like `backbone.blocks.N.mlp.w12.weight`:
```python
# Split fused w12 (shape: [2*hidden, embed]) into w1 and w2
w12 = tensor
half = w12.shape[0] // 2
w1 = w12[:half, :]  # gate
w2 = w12[half:, :]  # up
# Map to: backbone.blocks.N.mlp.w1.weight and backbone.blocks.N.mlp.w2.weight
```
Same for bias if present.

---

### BUG-3: SwiGLU hidden dimension is wrong

**File:** `sam3d_mlx/backbone.py:93`
**Severity:** CRITICAL — architecture dimensions mismatch, shapes won't load

PyTorch computes SwiGLU hidden dim with rounding:
```python
feedforward_channels = (int(feedforward_channels * 2 / 3) + 7) // 8 * 8
# For embed=1280, ratio=4.0: (int(5120 * 2/3) + 7) // 8 * 8 = 3416
```

MLX:
```python
hidden_dim = int(embed_dim * config.ffn_ratio)  # = 5120 — WRONG
```

**Fix:** In `backbone.py`, apply the same rounding formula:
```python
raw = int(embed_dim * config.ffn_ratio)
hidden_dim = (int(raw * 2 / 3) + 7) // 8 * 8  # = 3416
```

---

## High Severity (fix before PR if possible)

### BUG-4: rotmat_to_quat only handles trace > 0

**File:** `sam3d_mlx/mhr_utils.py:293-311`
**Severity:** HIGH — visible mesh distortion for extreme poses (~10-20% of frames)

Shepperd's method requires 4 branches (trace max, R[0,0] max, R[1,1] max, R[2,2] max). We only implement the `trace > 0` case. When trace is near -1 (joints near 180° rotation — elbows, shoulders in pitching), `sqrt(trace + 1)` approaches zero and division blows up despite the `1e-10` guard.

**Fix:** Implement all 4 branches of Shepperd's method, or use a continuous formulation. Reference: PyTorch uses `roma.unitquat_to_rotmat` which handles all cases. See scipy's `Rotation.from_matrix()` for a numpy-compatible implementation to port.

---

### BUG-5: CLIFF condition normalization wrong

**File:** `sam3d_mlx/batch_prep.py:185-215`
**Severity:** HIGH — wrong camera conditioning

PyTorch normalizes by focal length:
```python
condition_info[:, :2] = condition_info[:, :2] / focal_length
condition_info[:, 2] = condition_info[:, 2] / focal_length
```

MLX normalizes by image dimensions:
```python
cx_norm = (cx - W / 2.0) / (W / 2.0)
cy_norm = (cy - H / 2.0) / (H / 2.0)
```

These produce different values unless `focal_length == W/2`.

**Fix:** Match PyTorch — divide by focal length, not image dimensions.

---

## Medium Severity (document in PR, fix later)

### BUG-6: Ray conditioning uses stride sampling instead of bilinear+antialias

**File:** `sam3d_mlx/model.py:268`

PyTorch: `F.interpolate(rays, mode='bilinear', antialias=True)`
MLX: `rays[:, ::patch_size, ::patch_size, :]` (stride sampling)

Causes subtle ray-direction errors at image edges.

### BUG-7: LayerNorm doesn't upcast to float32

PyTorch's `LayerNorm32` upcasts to float32 before computing norm. MLX uses plain `nn.LayerNorm`. Matters if model runs in float16.

### BUG-8: mx.eval() called per-array

**File:** `sam3d_mlx/estimator.py:226-229`

Should batch: `mx.eval(a, b, c, d)` instead of 4 separate `mx.eval()` calls.

---

## Code Quality Issues (clean up for PR)

### Dead code in mhr_body.py
- `_apply_parameter_limits` lines 96-136: unused intermediate variables (`result_parts`, `all_indices`, `idx_expanded`)
- `_scatter_add_1d` lines 486-514: entire MLX attempt is dead code (loop body is `pass`), falls through to numpy

### numpy imports inside functions
- `mhr_head.py`: `import numpy as np` inside `_zero_at_indices` and `_scatter_set` — move to top of file

### All scatter operations bounce through numpy
- `_scatter_add_1d` converts to numpy for `np.add.at` — every skinning operation round-trips through CPU
- Performance concern for video pipeline but functionally correct

---

## mlx-vlm Integration Notes

For the PR to fit into Blaizzy/mlx-vlm, the following structural changes are needed:

1. `__init__.py` — export `Model`, `ModelConfig`, `VisionModel`, `LanguageModel`
2. Config — extend `BaseModelConfig` with `from_dict()`, add `model_type` field
3. Model class — rename `SAM3DBody` to `Model`, file to `sam3d_body.py`
4. Weight loading — replace custom `load_all_weights()` with static `sanitize()` method
5. `VisionModel` — wrap DINOv3Backbone; `LanguageModel` — stub class (SAM 3D Body has no text encoder)
6. Register `model_type: "sam3d_body"` in `MODEL_REMAPPING` in `utils.py`
7. Add `generate.py` with predictor interface matching SAM 3/3.1 pattern

---

## Fix Order

1. BUG-3 (hidden dim) — must fix first since it changes layer shapes
2. BUG-2 (w12 split) — weight conversion depends on correct shapes
3. BUG-1 (concat order) — simple swap
4. BUG-5 (CLIFF normalization) — simple formula change
5. BUG-4 (rotmat_to_quat) — most involved fix
6. Re-run weight conversion
7. Re-run inference on ohtani_ws_g7_pitch1.mp4
8. Compare output against PyTorch/MPS output
9. Clean up dead code
10. Restructure for mlx-vlm conventions
11. Open PR
