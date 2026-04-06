"""Stage-by-stage comparison of PyTorch JIT vs MLX MHR body model.

Feed the SAME model_params into both and compare outputs at each stage:
1. parameter_limits
2. parameter_transform (204 -> 889 joint DOFs)
3. blend_shapes (base + shape offsets)
4. pose_correctives (sparse -> dense -> vertex corrections)
5. forward_kinematics (joint DOFs -> global transforms)
6. linear_blend_skinning (transforms x rest verts -> posed verts)
"""

import sys
import numpy as np
import torch

sys.path.insert(0, "/Users/quartershots/Source/SamPlaysBaseball")

# --- Load test data ---
DATA_DIR = "data/output/813024"
pt_mp = np.load(f"{DATA_DIR}/pt_mhr_model_params.npy")  # (204,)
pt_verts_ref = np.load(f"{DATA_DIR}/pt_verts_frame178.npy")  # (18439, 3)

print(f"model_params: {pt_mp.shape}, range [{pt_mp.min():.4f}, {pt_mp.max():.4f}]")
print(f"pt_verts_ref: {pt_verts_ref.shape}, range [{pt_verts_ref.min():.4f}, {pt_verts_ref.max():.4f}]")

# --- Load PyTorch JIT model ---
print("\n=== Loading PyTorch JIT model ===")
jit_model = torch.jit.load("/tmp/sam3d-weights/assets/mhr_model.pt", map_location="cpu")
jit_model.eval()

# Inspect JIT model buffers
print("\nJIT model named buffers:")
jit_buffers = {}
for name, buf in jit_model.named_buffers():
    jit_buffers[name] = buf.numpy()
    if buf.numel() == 0:
        print(f"  {name}: {buf.shape} {buf.dtype} (empty)")
    elif buf.numel() < 1000:
        print(f"  {name}: {buf.shape} {buf.dtype} range=[{buf.min():.4f}, {buf.max():.4f}]")
    else:
        print(f"  {name}: {buf.shape} {buf.dtype}")

print("\nJIT model named parameters:")
jit_params = {}
for name, param in jit_model.named_parameters():
    jit_params[name] = param.detach().numpy()
    print(f"  {name}: {param.shape} {param.dtype}")

# --- Run PyTorch JIT model (black box) ---
print("\n=== Running PyTorch JIT forward ===")
with torch.no_grad():
    shape_zero = torch.zeros(1, 45)
    mp_torch = torch.from_numpy(pt_mp).unsqueeze(0).float()  # (1, 204)
    face_zero = torch.zeros(1, 72)

    pt_verts_out, pt_skel_out = jit_model(shape_zero, mp_torch, face_zero)
    pt_verts_jit = pt_verts_out.numpy()[0]  # (18439, 3)
    pt_skel_jit = pt_skel_out.numpy()[0]  # (127, 8)

print(f"JIT verts: {pt_verts_jit.shape}, range [{pt_verts_jit.min():.4f}, {pt_verts_jit.max():.4f}]")
print(f"JIT skel:  {pt_skel_jit.shape}")

# Compare JIT output to saved reference
diff_ref = np.abs(pt_verts_jit - pt_verts_ref).max()
print(f"\nJIT vs saved reference: max diff = {diff_ref:.6f}")
if diff_ref > 0.01:
    print("  WARNING: JIT output doesn't match saved reference!")

# --- Load MLX body model ---
print("\n=== Loading MLX body model ===")
import mlx.core as mx
from sam3d_mlx.mhr_body import MHRBodyModel, _scatter_add_1d
from sam3d_mlx.mhr_utils import euler_xyz_to_rotmat, quat_to_rotmat

bm = MHRBodyModel(num_joints=127, num_verts=18439)

# Load weights from safetensors
from safetensors.numpy import safe_open

MHR_KEY_MAP = {
    "character.skeleton.joint_translation_offsets": "joint_translation_offsets",
    "character.skeleton.joint_prerotations": "joint_prerotations",
    "character.skeleton.joint_parents": "joint_parents",
    "character.parameter_transform.parameter_transform": "parameter_transform",
    "character.parameter_transform.pose_parameters": "pose_parameters",
    "character.parameter_transform.rigid_parameters": "rigid_parameters",
    "character.parameter_transform.scaling_parameters": "scaling_parameters",
    "character.parameter_limits.minmax_min": "minmax_min",
    "character.parameter_limits.minmax_max": "minmax_max",
    "character.parameter_limits.minmax_weight": "minmax_weight",
    "character.parameter_limits.minmax_parameter_index": "minmax_parameter_index",
    "character.blend_shape.base_shape": "base_shape",
    "character.blend_shape.shape_vectors": "shape_vectors",
    "character.linear_blend_skinning.inverse_bind_pose": "inverse_bind_pose",
    "character.linear_blend_skinning.skin_indices_flattened": "skin_indices",
    "character.linear_blend_skinning.skin_weights_flattened": "skin_weights",
    "character.linear_blend_skinning.vert_indices_flattened": "vert_indices",
    "face_expressions.shape_vectors": "face_shape_vectors",
    "pose_correctives.pose_dirs_predictor.0.sparse_indices": "pc_sparse_indices",
    "pose_correctives.pose_dirs_predictor.0.sparse_weight": "pc_sparse_weight",
    "pose_correctives.pose_dirs_predictor.2.weight": "pc_linear_weight",
}

weights = []
with safe_open("/tmp/sam3d-mlx-weights/model.safetensors", framework="numpy") as f:
    for key in f.keys():
        if key.startswith("mhr."):
            mhr_key = key[len("mhr."):]
            mapped = MHR_KEY_MAP.get(mhr_key)
            if mapped is not None:
                tensor = mx.array(f.get_tensor(key))
                weights.append((mapped, tensor))

bm.load_weights(weights)
mx.eval(bm.parameters())
print(f"Loaded {len(weights)} MHR weight tensors into MLX body model")

# --- Stage 1: Parameter Limits ---
print("\n" + "=" * 60)
print("STAGE 1: Parameter Limits")
print("=" * 60)

mp_mx = mx.array(pt_mp[np.newaxis, :])  # (1, 204)

mlx_limited = bm._apply_parameter_limits(mp_mx)
mx.eval(mlx_limited)
mlx_limited_np = np.array(mlx_limited[0])

# PyTorch parameter limits using JIT model buffers
# The JIT model should have the same buffers
pt_minmax_min = jit_buffers.get("character_torch.parameter_limits.minmax_min",
                                 jit_buffers.get("character.parameter_limits.minmax_min"))
pt_minmax_max = jit_buffers.get("character_torch.parameter_limits.minmax_max",
                                 jit_buffers.get("character.parameter_limits.minmax_max"))
pt_minmax_idx = jit_buffers.get("character_torch.parameter_limits.minmax_parameter_index",
                                 jit_buffers.get("character.parameter_limits.minmax_parameter_index"))

if pt_minmax_min is not None:
    # Reproduce PyTorch parameter limits manually
    pt_param_vals = pt_mp[pt_minmax_idx.astype(int)]  # (198,)
    pt_clamped = np.clip(pt_param_vals, pt_minmax_min, pt_minmax_max)
    pt_limited = pt_mp.copy()
    for i in range(len(pt_minmax_idx)):
        pt_limited[int(pt_minmax_idx[i])] = pt_clamped[i]

    diff_limits = np.abs(mlx_limited_np - pt_limited).max()
    print(f"MLX vs PT parameter_limits: max diff = {diff_limits:.8f}")
    if diff_limits > 1e-5:
        divergent_idx = np.argmax(np.abs(mlx_limited_np - pt_limited))
        print(f"  First divergent index: {divergent_idx}")
        print(f"  MLX: {mlx_limited_np[divergent_idx]:.8f}, PT: {pt_limited[divergent_idx]:.8f}")
else:
    print("  Could not find parameter_limits buffers in JIT model")
    # List available buffer names
    print("  Available buffer names:")
    for name in sorted(jit_buffers.keys()):
        if "limit" in name.lower() or "minmax" in name.lower():
            print(f"    {name}")

# --- Stage 2: Parameter Transform ---
print("\n" + "=" * 60)
print("STAGE 2: Parameter Transform (204 -> 889 joint DOFs)")
print("=" * 60)

mlx_jdofs = bm._parameter_transform(mlx_limited)
mx.eval(mlx_jdofs)
mlx_jdofs_np = np.array(mlx_jdofs[0])

# PyTorch parameter transform
pt_PT_mat = jit_buffers.get("character_torch.parameter_transform.parameter_transform",
                             jit_buffers.get("character.parameter_transform.parameter_transform"))
if pt_PT_mat is not None:
    # Use the limited params from stage 1
    if pt_minmax_min is not None:
        pt_input = pt_limited
    else:
        pt_input = pt_mp
    pt_padded = np.concatenate([pt_input, np.zeros(249 - 204)])  # (249,)
    pt_jdofs = pt_padded @ pt_PT_mat.T  # (889,)

    diff_jdofs = np.abs(mlx_jdofs_np - pt_jdofs).max()
    print(f"MLX vs PT joint_dofs: max diff = {diff_jdofs:.8f}")
    print(f"  MLX range: [{mlx_jdofs_np.min():.4f}, {mlx_jdofs_np.max():.4f}]")
    print(f"  PT  range: [{pt_jdofs.min():.4f}, {pt_jdofs.max():.4f}]")

    if diff_jdofs > 1e-4:
        divergent_idx = np.argmax(np.abs(mlx_jdofs_np - pt_jdofs))
        print(f"  First divergent DOF index: {divergent_idx}")
        print(f"  MLX: {mlx_jdofs_np[divergent_idx]:.8f}, PT: {pt_jdofs[divergent_idx]:.8f}")
else:
    print("  Could not find parameter_transform in JIT model")

# --- Stage 3: Blend Shapes ---
print("\n" + "=" * 60)
print("STAGE 3: Blend Shapes")
print("=" * 60)

shape_zero_mx = mx.zeros((1, 45))
face_zero_mx = mx.zeros((1, 72))
mlx_bshape_verts = bm._blend_shapes(shape_zero_mx, face_zero_mx)
mx.eval(mlx_bshape_verts)
mlx_bshape_np = np.array(mlx_bshape_verts[0])

# With zero shape and face params, blend shapes should just be base_shape
pt_base_shape = jit_buffers.get("character_torch.blend_shape.base_shape",
                                 jit_buffers.get("character.blend_shape.base_shape"))
if pt_base_shape is not None:
    diff_bshape = np.abs(mlx_bshape_np - pt_base_shape).max()
    print(f"MLX vs PT base_shape (zero params): max diff = {diff_bshape:.8f}")
    print(f"  MLX range: [{mlx_bshape_np.min():.4f}, {mlx_bshape_np.max():.4f}]")
    print(f"  PT  range: [{pt_base_shape.min():.4f}, {pt_base_shape.max():.4f}]")
else:
    print("  Could not find base_shape in JIT model")

# --- Stage 4: Pose Correctives ---
print("\n" + "=" * 60)
print("STAGE 4: Pose Correctives")
print("=" * 60)

# Use the PT joint_dofs from stage 2 (or MLX if PT unavailable)
if pt_PT_mat is not None:
    # Feed PT joint_dofs into MLX pose correctives to isolate this stage
    jdofs_mx = mx.array(pt_jdofs[np.newaxis, :].astype(np.float32))
else:
    jdofs_mx = mlx_jdofs

mlx_corrections = bm._pose_correctives(jdofs_mx, 18439)
mx.eval(mlx_corrections)
mlx_corrections_np = np.array(mlx_corrections[0])

# PyTorch pose correctives using JIT buffers
pc_sparse_idx = jit_buffers.get("pose_correctives_model.pose_dirs_predictor.0.sparse_indices")
pc_sparse_wt = jit_buffers.get("pose_correctives_model.pose_dirs_predictor.0.sparse_weight")
pc_linear_wt = jit_params.get("pose_correctives_model.pose_dirs_predictor.2.weight")

if pc_sparse_idx is not None and pc_sparse_wt is not None and pc_linear_wt is not None:
    # Sparse layer
    jdofs_input = pt_jdofs if pt_PT_mat is not None else mlx_jdofs_np
    out_idx = pc_sparse_idx[0].astype(int)  # output indices
    in_idx = pc_sparse_idx[1].astype(int)   # input indices

    input_vals = jdofs_input[in_idx]
    weighted = input_vals * pc_sparse_wt

    sparse_out = np.zeros(3000, dtype=np.float32)
    np.add.at(sparse_out, out_idx, weighted)

    # ReLU
    sparse_out = np.maximum(sparse_out, 0)

    # Dense layer
    dense_out = sparse_out @ pc_linear_wt.T  # (55317,)
    pt_corrections = dense_out.reshape(18439, 3)

    diff_corrections = np.abs(mlx_corrections_np - pt_corrections).max()
    print(f"MLX vs PT pose_correctives: max diff = {diff_corrections:.8f}")
    print(f"  MLX range: [{mlx_corrections_np.min():.6f}, {mlx_corrections_np.max():.6f}]")
    print(f"  PT  range: [{pt_corrections.min():.6f}, {pt_corrections.max():.6f}]")
    print(f"  MLX L2 norm: {np.linalg.norm(mlx_corrections_np):.4f}")
    print(f"  PT  L2 norm: {np.linalg.norm(pt_corrections):.4f}")

    if diff_corrections > 1e-3:
        # Check sub-stages
        mlx_sparse_check = mx.array(jdofs_input[np.newaxis, :].astype(np.float32))
        mx.eval(mlx_sparse_check)

        # Check sparse output
        mlx_in_vals = np.array(jdofs_input[in_idx])
        mlx_weighted = mlx_in_vals * np.array(bm.pc_sparse_weight)
        mlx_sparse_out = np.zeros(3000, dtype=np.float32)
        np.add.at(mlx_sparse_out, np.array(bm.pc_sparse_indices[0]).astype(int), mlx_weighted)

        diff_sparse = np.abs(mlx_sparse_out - sparse_out).max()
        print(f"\n  Sub-check - sparse layer: max diff = {diff_sparse:.8f}")

        # Check after ReLU
        mlx_relu = np.maximum(mlx_sparse_out, 0)
        diff_relu = np.abs(mlx_relu - sparse_out).max()
        print(f"  Sub-check - after ReLU:   max diff = {diff_relu:.8f}")
else:
    print("  Could not find pose_correctives in JIT model")
    print("  Available buffer/param names with 'pose':")
    for name in sorted(list(jit_buffers.keys()) + list(jit_params.keys())):
        if "pose" in name.lower():
            print(f"    {name}")

# --- Stage 5: Forward Kinematics ---
print("\n" + "=" * 60)
print("STAGE 5: Forward Kinematics")
print("=" * 60)

# Use the same joint_dofs for both
if pt_PT_mat is not None:
    fk_jdofs_mx = mx.array(pt_jdofs[np.newaxis, :].astype(np.float32))
else:
    fk_jdofs_mx = mlx_jdofs

mlx_skel, mlx_gpos, mlx_grot, mlx_gscale = bm._forward_kinematics(fk_jdofs_mx)
mx.eval(mlx_skel, mlx_gpos, mlx_grot, mlx_gscale)
mlx_gpos_np = np.array(mlx_gpos[0])  # (127, 3)
mlx_grot_np = np.array(mlx_grot[0])  # (127, 3, 3)
mlx_skel_np = np.array(mlx_skel[0])  # (127, 8)

# Compare against JIT skel_state
diff_skel = np.abs(mlx_skel_np[:, :3] - pt_skel_jit[:, :3]).max()
diff_skel_quat = np.abs(mlx_skel_np[:, 3:7] - pt_skel_jit[:, 3:7]).max()
diff_skel_scale = np.abs(mlx_skel_np[:, 7:8] - pt_skel_jit[:, 7:8]).max()
print(f"MLX vs JIT skel_state positions: max diff = {diff_skel:.6f}")
print(f"MLX vs JIT skel_state quaternions: max diff = {diff_skel_quat:.6f}")
print(f"MLX vs JIT skel_state scales: max diff = {diff_skel_scale:.6f}")

# Find worst joint
joint_pos_diffs = np.linalg.norm(mlx_skel_np[:, :3] - pt_skel_jit[:, :3], axis=1)
worst_joint = np.argmax(joint_pos_diffs)
print(f"\nWorst joint position error: joint {worst_joint}, diff = {joint_pos_diffs[worst_joint]:.6f}")
print(f"  MLX pos: {mlx_skel_np[worst_joint, :3]}")
print(f"  PT  pos: {pt_skel_jit[worst_joint, :3]}")

# Check root joint specifically
print(f"\nRoot joint (0):")
print(f"  MLX: pos={mlx_skel_np[0, :3]}, quat={mlx_skel_np[0, 3:7]}, scale={mlx_skel_np[0, 7]}")
print(f"  PT:  pos={pt_skel_jit[0, :3]}, quat={pt_skel_jit[0, 3:7]}, scale={pt_skel_jit[0, 7]}")

# --- Stage 6: Linear Blend Skinning ---
print("\n" + "=" * 60)
print("STAGE 6: Linear Blend Skinning (final output)")
print("=" * 60)

# Feed the exact same rest verts + FK results to LBS
rest_verts_mx = mlx_bshape_verts + mlx_corrections
mx.eval(rest_verts_mx)

mlx_skinned = bm._linear_blend_skinning(rest_verts_mx, mlx_gpos, mlx_grot, mlx_gscale)
mx.eval(mlx_skinned)
mlx_skinned_np = np.array(mlx_skinned[0])

diff_final = np.abs(mlx_skinned_np - pt_verts_jit).max()
diff_final_mean = np.abs(mlx_skinned_np - pt_verts_jit).mean()
diff_final_mm = diff_final * 1000  # assume meters

print(f"MLX vs JIT final vertices: max diff = {diff_final:.6f} ({diff_final_mm:.1f}mm)")
print(f"MLX vs JIT final vertices: mean diff = {diff_final_mean:.6f}")
print(f"  MLX range: [{mlx_skinned_np.min():.4f}, {mlx_skinned_np.max():.4f}]")
print(f"  PT  range: [{pt_verts_jit.min():.4f}, {pt_verts_jit.max():.4f}]")

# Vertex-wise L2 distance
vert_dists = np.linalg.norm(mlx_skinned_np - pt_verts_jit, axis=1)
print(f"\n  Per-vertex L2: mean={vert_dists.mean():.6f}, max={vert_dists.max():.6f}")
print(f"  Vertices >1mm: {(vert_dists > 0.001).sum()} / {len(vert_dists)}")
print(f"  Vertices >10mm: {(vert_dists > 0.01).sum()} / {len(vert_dists)}")
print(f"  Vertices >100mm: {(vert_dists > 0.1).sum()} / {len(vert_dists)}")

worst_vert = np.argmax(vert_dists)
print(f"\n  Worst vertex: {worst_vert}, diff = {vert_dists[worst_vert]:.6f}")
print(f"  MLX: {mlx_skinned_np[worst_vert]}")
print(f"  PT:  {pt_verts_jit[worst_vert]}")

# --- Summary ---
print("\n" + "=" * 60)
print("SUMMARY — First stage with significant divergence")
print("=" * 60)

stages = []
if pt_minmax_min is not None:
    stages.append(("1. Parameter Limits", diff_limits))
if pt_PT_mat is not None:
    stages.append(("2. Parameter Transform", diff_jdofs))
if pt_base_shape is not None:
    stages.append(("3. Blend Shapes", diff_bshape))
if pc_linear_wt is not None:
    stages.append(("4. Pose Correctives", diff_corrections))
stages.append(("5. FK - positions", diff_skel))
stages.append(("5. FK - quaternions", diff_skel_quat))
stages.append(("5. FK - scales", diff_skel_scale))
stages.append(("6. LBS (final)", diff_final))

for name, diff in stages:
    flag = " <-- BUG HERE?" if diff > 0.01 else (" ~" if diff > 1e-4 else " ✓")
    print(f"  {name}: {diff:.8f}{flag}")
