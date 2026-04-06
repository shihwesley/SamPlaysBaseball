# Height-Based Scale Correction + Mesh Quality Investigation

## 1. Height-Based Scale Correction (Baseball App Feature)

### Problem
SAM 3D Body outputs meshes in a canonical SMPL coordinate space. The mesh height (~1.76m in our tests) is the model's best guess from monocular vision, but it can't distinguish a 6'4" pitcher from a 5'10" one — that's an inherent monocular ambiguity.

For biomechanics analysis (stride length, arm extension, release point height), we need physically accurate distances, not canonical-space approximations.

### Solution
Use MLB roster data (height, weight) to post-process the mesh output.

**Step 1 — Scale correction:**
```python
# Player height from MLB Stats API (already in our pitch_db)
actual_height_m = 1.93  # Ohtani: 6'4"
mesh_height_m = mesh_span_y  # ~1.76m from model output

scale_factor = actual_height_m / mesh_height_m
vertices_scaled = vertices * scale_factor
camera_t_scaled = camera_t * scale_factor  # keep projection consistent
```

This gives physically accurate distances: stride length in meters, release point height in meters, arm extension in meters.

**Step 2 — Weight-informed shape prior (future):**
SMPL beta[0] correlates with height, beta[1] with weight/build. Could constrain betas using roster data:
```
height=193cm, weight=95kg -> beta[0] ≈ 2.1, beta[1] ≈ 0.8
```
This would improve body proportions (limb length ratios), not just overall scale.

### Where to add
- `sam3d_mlx/estimator.py` or a new `scripts/scale_correction.py`
- NOT in the mlx-vlm port (this is baseball-app-specific)
- Player data from `backend/app/data/player_search.py` (already has MLB Stats API integration)

### Implementation cost
~20 lines for scale correction. Shape prior is more involved (need a height/weight→beta regression model or lookup table).

---

## 2. Mesh Quality Issue — Near-Neutral Pose

### Symptom
MLX mesh output shows a near-neutral standing pose (arms at sides, legs straight) regardless of the pitcher's actual motion. PyTorch/MPS output on similar frames shows dynamic posing (arm raised, legs apart, matching the real pose).

The camera placement IS correct — the mesh appears in the right location on the pitcher.

### What we've ruled out
- **Backbone features:** Healthy. Range [-5.5, 4.5], std=0.30, 99.7% non-zero. The vision encoder is working.
- **MHR body model buffers:** All loaded correctly. base_shape, skin_weights, joint_prerotations, pose_correctives — all have non-zero data with plausible ranges.
- **Concat order:** Verified correct via weight column norms. The first 3 columns of init_to_token_mhr.weight have 2-3x larger norms (2.6, 1.6, 1.3 vs mean 0.81), confirming they expect the CLIFF condition vector (which has larger magnitude values).
- **Camera head:** Producing reasonable params (scale, tx, ty track correctly across frames).

### Where the signal drops
The 204-dim model_params (pose head output that drives the body model) have:
- `std=0.40`, nearly identical to `init_pose std=0.40`
- `97/204 params near zero`
- The decoder's iterative refinement isn't moving params far enough from the learnable init

This means the decoder transformer IS running but its pose token output isn't carrying enough pose signal to drive meaningful body deformation.

### Likely culprits (in order of suspicion)

**A. CLIFF normalization change (BUG-5 fix)**
We changed from `(cx - W/2) / (W/2)` to `(cx - W/2) / focal_length`. For our test frame:
- Old: cx_norm ≈ -0.15, cy_norm ≈ 0.21, crop_ratio ≈ 0.49
- New: depends on focal_length (which varies with image height and FOV assumption)

If the model was actually trained with the image-dimension normalization (not focal_length), our "fix" is feeding wrong conditioning. The PyTorch code we referenced might be from a different checkpoint version.

**Action:** Test inference with the original normalization restored.

**B. Ray conditioning change (BUG-6 fix)**
We changed from stride sampling (`rays[:, ::16, ::16, :]`) to `AvgPool2d(16, 16)`. For smoothly varying ray directions, these are similar but not identical. The ray features are Fourier-encoded and concatenated with backbone features — subtle differences get amplified by the decoder attention.

**Action:** Test inference with stride sampling restored.

**C. LayerNorm32 change (BUG-7 fix)**
We added float32 upcast to LayerNorm in the decoder. If the model was trained with plain LayerNorm (no upcast), and our weights are in float16, the upcast could change intermediate values enough to shift decoder behavior.

**Action:** Test inference with plain nn.LayerNorm restored.

**D. Decoder numerical accumulation**
6 decoder layers of self-attention + cross-attention + FFN. Small per-layer differences (from LayerNorm, ray features, or conditioning) accumulate. By layer 6, the pose token could be significantly different from the PyTorch output.

**Action:** Compare per-layer decoder output between MLX and PyTorch.

### Recommended debug sequence
1. Revert BUG-5 (CLIFF normalization) → re-run inference → check mesh pose
2. If no improvement, revert BUG-6 (ray downsampling) → re-run
3. If no improvement, revert BUG-7 (LayerNorm32) → re-run
4. If still stuck, run PyTorch on the same frame and compare decoder token outputs layer by layer

### Note for the PR
The PR to mlx-vlm should document this as a known issue:
> "Mesh deformation may appear less dynamic than PyTorch reference. Camera parameters and mesh scale are correct. Pose refinement quality under investigation."

The model is still useful — it produces valid meshes at correct positions with correct proportions. The pose accuracy gap is a refinement issue, not a broken pipeline.

---

## 3. Debug Findings (2026-04-05)

### Part 1 — Height Scaling: DONE ✅

Implemented in `backend/app/data/player_search.py`, `backend/app/models/pitch.py`, `backend/app/pipeline/inference.py`:
- `_parse_height_to_meters()` — parses MLB API strings like `"6' 4\""` to meters
- `Player` dataclass extended with `height_m` and `weight_lbs`
- `PitchMetadata.player_height_m` carries roster height through pipeline
- `PitchData.scale_to_height()` applies uniform scale: `actual_height / mesh_y_span`
- Wired into `process_video_frames()` — no-ops when height is None
- Validated: Ohtani 6'4" → scale factor 1.25x, shoulder width 44.6cm ✓

### Part 2 — Mesh Quality: Root Cause Found

#### What we ruled out

| Suspect | Status | Evidence |
|---------|--------|----------|
| BUG-5: CLIFF normalization | ❌ Ruled out | Reverting produces identical output (0.0mm vertex delta) |
| BUG-6: Ray downsampling (AvgPool vs stride) | ❌ Ruled out | Reverting produces identical output (0.0mm vertex delta) |
| BUG-7: LayerNorm32 upcast | ❌ Ruled out | MLX runs in float32 natively — upcast is a no-op. 0 instances found to replace. |
| Decoder pose params too weak | ❌ Ruled out | With matched bbox, MLX model_params L2 delta from PyTorch is only 0.93. Body pose std is 0.37 (both). |
| Degenerate faces / collapsed mesh | ❌ Misdiagnosed | Threshold of 1e-4 m² was too aggressive for a 18439-vertex mesh. Real degenerate count at 1e-6 is ~29 (normal). |
| Bounding box mismatch | ⚠️ Was confounding | PyTorch `process_one_image()` without bboxes defaults to full frame `[0,0,1280,720]`. MLX detects tight crop `[681,189,789,452]`. This caused earlier comparison to show 170° global rotation difference. **With matched bbox, global rot matches within 0.1 rad.** |

#### Actual root cause: MLX MHR body model produces wrong vertices

**Evidence:** With the same detected bbox passed to both pipelines:
- `mhr_model_params` L2 delta: **0.93** (close)
- Global rotation: PT `[2.98, -0.55, 3.02]` vs MLX `[2.94, -0.43, 3.11]` (~6° difference)
- Body pose std: PT 0.37 vs MLX 0.37 (identical)
- **But vertex delta: mean 2574mm, max 3360mm — every vertex diverges by >10mm**

The decoder is producing similar parameters. The body model is producing wildly different vertices from those parameters. The bug is in the MLX `MHRBodyModel` implementation (`sam3d_mlx/mhr_body.py`).

### Part 2 — What to debug next: MHR Body Model

The MLX body model pipeline is:

```
model_params (204) → parameter_limits → parameter_transform (889 joint DOFs)
                                       → blend_shapes (rest verts + shape)
                                       → pose_correctives (joint_dofs → vertex corrections)
                                       → forward_kinematics (joint DOFs → global transforms)
                                       → linear_blend_skinning (transforms × rest verts → posed verts)
```

#### Step-by-step comparison needed

Feed the **same model_params** (from PyTorch) into both body models and compare outputs at each stage:

**1. Parameter limits (`_apply_parameter_limits`)**
- MLX uses a loop-based scatter for clamping (lines 96-104). Verify the clamped output matches PyTorch.
- Concern: The loop-based `_set_column` replacement may have off-by-one or overwrite issues.

**2. Parameter transform (`_parameter_transform`)**
- Matrix multiply: `padded (B, 249) @ PT.T (249, 889)`. Should be straightforward.
- Compare the 889-dim `joint_dofs` output.

**3. Blend shapes (`_blend_shapes`)**
- `base_shape + einsum('bs,svd->bvd', shape, shape_vectors)`. Simple.
- Compare rest-pose vertices after blend shapes.

**4. Pose correctives (`_pose_correctives`)**
- **Likely suspect.** Uses:
  - Sparse matrix multiply via `_scatter_add_1d` (GPU→CPU→GPU round-trip)
  - ReLU activation
  - Dense linear `(55317, 3000)` 
  - Reshape to `(18439, 3)`
- The sparse scatter-add uses numpy (`np.add.at`) — verify indices and values match PyTorch's sparse layer output.
- The sparse→dense→reshape chain has many places for dimension/index errors.

**5. Forward kinematics (`_forward_kinematics`)**
- **Likely suspect.** Iterates 127 joints with:
  - `euler_xyz_to_rotmat` — verify rotation convention matches PyTorch (XYZ vs ZYX vs intrinsic vs extrinsic)
  - `quat_to_rotmat` for prerotations — verify quaternion convention (w-first vs w-last, Hamilton vs JPL)
  - Prerotation composition: `prerot @ local_rot` — verify multiplication order
  - FK chain: `parent_pos + parent_scale * (parent_rot @ local_trans)` — verify transform order
- Key question: does PyTorch's JIT model use the same euler convention? The `euler_xyz_to_rotmat` in `mhr_utils.py` must exactly match.

**6. Linear blend skinning (`_linear_blend_skinning`)**
- **Likely suspect.** Key operations:
  - `combined_rot = einsum('bjik,jkl->bjil', global_rot, ibp_rot)` — verify einsum contraction order
  - `combined_trans = global_pos + einsum('bjik,jk->bji', global_rot, ibp_trans) * global_scale` — verify scale application
  - Per-vertex transform: `cs * einsum('bnij,bnj->bni', cr, v) + ct` — verify transform order
  - Scatter-add via `_scatter_add_1d` (GPU→CPU→GPU round-trip) — verify accumulation correctness
- The einsum contractions in LBS are the most likely place for a transposition bug — one wrong index pair flips the rotation.

### Part 2 — FIXED: Three Bugs in MLX Body Model (2026-04-05)

**Stage-by-stage comparison** (scripts/compare_body_model_stages.py, scripts/debug_fk_chain.py):

| Stage | Before Fix | After Fix |
|-------|-----------|-----------|
| 1. Parameter Limits | N/A (was applied incorrectly) | Removed |
| 2. Parameter Transform | ✅ matched | ✅ matched |
| 3. Blend Shapes | ✅ matched | ✅ matched |
| 4. Pose Correctives | Wrong input (889D vs 750D) | max diff 0.0000008 |
| 5. Forward Kinematics | 17mm error | max diff 0.00003mm |
| 6. LBS (final verts) | 17,000mm error | max diff 0.0001mm |

#### Bug 1: Parameter limits applied at inference (the big one)

`mhr_body.py:__call__` called `_apply_parameter_limits()` before parameter transform.
The JIT model never calls parameter_limits in its forward pass — it's for training constraints only.
This clamped 33 of 204 parameters (some to zero), causing ~17mm FK position errors.

**Fix:** Removed the `_apply_parameter_limits` call from `__call__`.

#### Bug 2: Wrong scale formula

MLX used `1 + dof`, PyTorch uses `exp(dof × ln(2)) = 2^dof`.
Small impact (only 2 joints had non-zero scale DOFs), but semantically wrong.

**Fix:** Changed to `mx.exp(local_scale * 0.6931471824645996)`.

#### Bug 3: Wrong pose correctives input format

MLX fed raw 889D joint DOFs to the sparse predictor. PyTorch extracts 6D rotation
features from euler angles of joints 2-126 (750D = 125 × 6), subtracts identity
rotation, then feeds to the predictor. The sparse layer indices only go up to 749.

**Fix:** Added `_pose_features_from_joint_dofs()` that matches PyTorch's preprocessing.

#### Rotation convention checklist (verified correct)

- [x] Euler angle order: XYZ (extrinsic) = ZYX (intrinsic) — R = Rz @ Ry @ Rx
- [x] Quaternion layout: `[x, y, z, w]` (xyzw, confirmed by near-identity joints)
- [x] Prerotation composition: `prerot @ local` (Hamilton multiply order)
- [x] FK transform order: `parent_rot @ child_trans` ✓

All conventions were correct — the bugs were algorithmic, not convention mismatches.
| `data/output/813024/inn2_ab11_p5_FF_keypoints_scaled.npy` | MLX keypoints scaled (121, 70, 3) |
