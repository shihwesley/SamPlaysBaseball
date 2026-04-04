---
name: mlx-mhr-head
phase: 2
sprint: 2
parent: mlx-port-manifest
depends_on: [mlx-decoder]
status: draft
created: 2026-04-03
---

# MLX MHR Head — Body Model + Forward Kinematics

Port the MHR (Meta Human Recovery) pose head from JIT TorchScript to pure MLX.
This is the biggest rewrite — replacing a TorchScript body model with native MLX math.

## Requirements

- Port MHRHead: proj FFN → pose parameter regression → MHR forward
- Port MHR forward kinematics: joint_parameters → skeleton_state → vertices
- Port blend shape evaluation (shape + expression parameters → vertex offsets)
- Port linear blend skinning (skeleton pose → skinned vertices)
- Port parameter limits/constraints (joint angle limits)
- Port scale/shape decomposition (PCA space → actual body proportions)
- Port hand pose replacement logic (PCA hand → model params)
- No TorchScript or PyTorch dependency

## Acceptance Criteria

- [ ] Given same pose/shape/scale params, produces same vertices within 1e-3
- [ ] Joint coordinates match PyTorch within 1e-3
- [ ] Skinned mesh (18,439 verts) matches PyTorch within 1e-3
- [ ] Forward kinematics chain produces correct global rotations
- [ ] MHR head inference < 50ms on M-series

## Architecture

The MHR body model implements a character skeleton pipeline:

```
Input: shape_params (45), model_params (136+68), expr_params (72)
  ↓
1. Blend shapes: shape_params → vertex offsets (base_shape + shape_vectors)
  ↓
2. Parameter transform: model_params → joint local rotations
  ↓
3. Forward kinematics: local rotations → global skeleton state
   (traverse parent chain, accumulate rotations + translations)
  ↓
4. Linear blend skinning: skeleton + rest vertices → posed vertices
  ↓
5. Pose correctives: joint angles → additional vertex displacements
  ↓
Output: (18439 vertices, 127-joint skeleton state)
```

### JIT Model Buffer Extraction

The TorchScript model contains these buffers (from convert_weights):
- `skeleton.joint_translation_offsets` (127, 3)
- `skeleton.joint_prerotations` (127, 4) — quaternions
- `skeleton.joint_parents` (127,) — parent indices
- `mesh.rest_vertices` (18439, 3)
- `mesh.faces` (36874, 3)
- `linear_blend_skinning.inverse_bind_pose` (127, 4, 4)
- `linear_blend_skinning.skin_weights_flattened` + `skin_indices_flattened`
- `blend_shape.shape_vectors` (45, 18439*3) — PCA basis
- `blend_shape.base_shape` (18439*3,)
- `parameter_transform.*` — joint param mapping
- `parameter_limits.*` — angle constraints

### MLX FK Implementation

Key operation: traverse skeleton tree from root to leaves, accumulate transforms.
In MLX this is a sequential loop over 127 joints (can't parallelize due to parent deps).
This is the same bottleneck as in PyTorch — the FK chain is inherently sequential.

```python
# Pseudocode for FK in MLX
global_transforms = mx.zeros((B, 127, 4, 4))
for j in range(127):
    parent = parents[j]
    local_T = compose_transform(rotations[j], translations[j])
    if parent == -1:
        global_transforms[:, j] = local_T
    else:
        global_transforms[:, j] = global_transforms[:, parent] @ local_T
```

## Files

| File | Action |
|------|--------|
| `sam3d_mlx/mhr_head.py` | create (MHRHead with pose regression) |
| `sam3d_mlx/mhr_body.py` | create (FK, skinning, blend shapes) |
| `sam3d_mlx/mhr_utils.py` | create (rotation math: roma replacements, body/hand continuous-to-euler conversion, index arrays) |
| `tests/test_mhr.py` | create (parity test) |

## Tasks

1. Extract all buffers from JIT model, map to safetensors keys
2. Port roma rotation math to MLX: rotmat_to_euler, euler_to_rotmat, unitquat_to_rotmat
3. Port compact_cont_to_model_params_body (260-dim continuous → 133 euler, with hardcoded 23×3-DOF + 58×1-DOF index arrays)
4. Port compact_cont_to_model_params_hand (54-dim → 27 euler per hand) + batch6DFromXYZ/batchXYZfrom6D
5. Implement blend shape evaluation in MLX
6. Implement depth-batched forward kinematics (group joints by skeleton tree depth, batch matmul per level instead of 127 sequential ops)
7. Implement linear blend skinning (sparse weight * transform)
8. Implement MHRHead (proj FFN + mhr_forward + replace_hands_in_pose)
9. Write unit tests for all conversion functions: roundtrip parity for cont-to-euler, index array coverage verification (all 133 params covered, no overlap)
10. Write per-stage parity tests (blend shapes, FK, skinning, full pipeline)
