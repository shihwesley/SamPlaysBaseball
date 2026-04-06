"""Deep FK comparison: trace joint-by-joint where MLX diverges from PyTorch.

We know stages 1-3 match. The bug is in Forward Kinematics.
This script:
1. Extracts the PyTorch FK implementation from the JIT model by running it
   and capturing the skel_state output
2. Reimplements FK step-by-step in numpy matching the PyTorch JIT conventions
3. Compares MLX FK against numpy FK at each joint
"""

import sys
import numpy as np
import torch

sys.path.insert(0, "/Users/quartershots/Source/SamPlaysBaseball")

# --- Load test data and models ---
DATA_DIR = "data/output/813024"
pt_mp = np.load(f"{DATA_DIR}/pt_mhr_model_params.npy")  # (204,)

# Load JIT model
jit_model = torch.jit.load("/tmp/sam3d-weights/assets/mhr_model.pt", map_location="cpu")
jit_model.eval()

# Get JIT buffers
jit_buffers = {}
for name, buf in jit_model.named_buffers():
    jit_buffers[name] = buf.numpy() if buf.numel() > 0 else np.array([])

# Run JIT to get ground truth
with torch.no_grad():
    shape_zero = torch.zeros(1, 45)
    mp_torch = torch.from_numpy(pt_mp).unsqueeze(0).float()
    face_zero = torch.zeros(1, 72)
    pt_verts, pt_skel = jit_model(shape_zero, mp_torch, face_zero)
    pt_skel_np = pt_skel.numpy()[0]  # (127, 8) = [x,y,z,qx,qy,qz,qw,scale]

# Get buffers
joint_offsets = jit_buffers["character_torch.skeleton.joint_translation_offsets"]  # (127, 3)
joint_prerot = jit_buffers["character_torch.skeleton.joint_prerotations"]  # (127, 4) quaternions
joint_parents = jit_buffers["character_torch.skeleton.joint_parents"]  # (127,) int32
PT_mat = jit_buffers["character_torch.parameter_transform.parameter_transform"]  # (889, 249)
minmax_min = jit_buffers["character_torch.parameter_limits.minmax_min"]
minmax_max = jit_buffers["character_torch.parameter_limits.minmax_max"]
minmax_idx = jit_buffers["character_torch.parameter_limits.minmax_parameter_index"]

# Reproduce stages 1-2 in numpy
# Stage 1: Parameter limits
limited = pt_mp.copy()
for i in range(len(minmax_idx)):
    idx = int(minmax_idx[i])
    limited[idx] = np.clip(limited[idx], minmax_min[i], minmax_max[i])

# Stage 2: Parameter transform
padded = np.concatenate([limited, np.zeros(249 - 204)])
joint_dofs = padded @ PT_mat.T  # (889,)
jd = joint_dofs.reshape(127, 7)

print(f"Joint DOFs: shape={jd.shape}, range=[{jd.min():.4f}, {jd.max():.4f}]")
print(f"  Per-joint DOF layout: [tx, ty, tz, rx, ry, rz, scale]")

# --- Check quaternion convention ---
print("\n=== Quaternion Convention Check ===")
print(f"Joint prerotations shape: {joint_prerot.shape}")
print(f"First few prerotations:")
for j in range(5):
    q = joint_prerot[j]
    norm = np.linalg.norm(q)
    print(f"  Joint {j}: [{q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f}] norm={norm:.6f}")

# Check if quaternions are [x,y,z,w] or [w,x,y,z]
# If [x,y,z,w], identity quaternion should have last element = 1
# If [w,x,y,z], identity quaternion should have first element = 1
# Look for joints with near-identity prerotation
near_identity = []
for j in range(127):
    q = joint_prerot[j]
    # Check [x,y,z,w] convention: last element close to ±1
    if abs(abs(q[3]) - 1.0) < 0.01:
        near_identity.append(("xyzw", j, q))
    # Check [w,x,y,z] convention: first element close to ±1
    if abs(abs(q[0]) - 1.0) < 0.01:
        near_identity.append(("wxyz", j, q))

print(f"\nJoints with near-identity prerotation:")
for conv, j, q in near_identity[:10]:
    print(f"  Convention={conv}, Joint {j}: {q}")


# --- NumPy reference FK implementations ---
def quat_to_rotmat_xyzw(q):
    """Quaternion [x,y,z,w] to rotation matrix."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    x2, y2, z2 = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.zeros(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2*(y2 + z2)
    R[..., 0, 1] = 2*(xy - wz)
    R[..., 0, 2] = 2*(xz + wy)
    R[..., 1, 0] = 2*(xy + wz)
    R[..., 1, 1] = 1 - 2*(x2 + z2)
    R[..., 1, 2] = 2*(yz - wx)
    R[..., 2, 0] = 2*(xz - wy)
    R[..., 2, 1] = 2*(yz + wx)
    R[..., 2, 2] = 1 - 2*(x2 + y2)
    return R


def quat_to_rotmat_wxyz(q):
    """Quaternion [w,x,y,z] to rotation matrix."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    x2, y2, z2 = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.zeros(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2*(y2 + z2)
    R[..., 0, 1] = 2*(xy - wz)
    R[..., 0, 2] = 2*(xz + wy)
    R[..., 1, 0] = 2*(xy + wz)
    R[..., 1, 1] = 1 - 2*(x2 + z2)
    R[..., 1, 2] = 2*(yz - wx)
    R[..., 2, 0] = 2*(xz - wy)
    R[..., 2, 1] = 2*(yz + wx)
    R[..., 2, 2] = 1 - 2*(x2 + y2)
    return R


def euler_xyz_to_rotmat(angles):
    """XYZ euler -> rotation matrix, R = Rz @ Ry @ Rx (extrinsic XYZ = intrinsic ZYX)."""
    cx = np.cos(angles[..., 0])
    sx = np.sin(angles[..., 0])
    cy = np.cos(angles[..., 1])
    sy = np.sin(angles[..., 1])
    cz = np.cos(angles[..., 2])
    sz = np.sin(angles[..., 2])
    R = np.zeros(angles.shape[:-1] + (3, 3))
    R[..., 0, 0] = cz * cy
    R[..., 0, 1] = cz * sy * sx - sz * cx
    R[..., 0, 2] = cz * sy * cx + sz * sx
    R[..., 1, 0] = sz * cy
    R[..., 1, 1] = sz * sy * sx + cz * cx
    R[..., 1, 2] = sz * sy * cx - cz * sx
    R[..., 2, 0] = -sy
    R[..., 2, 1] = cy * sx
    R[..., 2, 2] = cy * cx
    return R


# --- Try different FK conventions ---
def run_fk(jd, joint_offsets, joint_prerot, joint_parents,
           quat_convention="xyzw", prerot_order="prerot_local"):
    """Run FK with specified conventions.

    Args:
        quat_convention: "xyzw" or "wxyz" for prerotation quaternions
        prerot_order: "prerot_local" (prerot @ local) or "local_prerot" (local @ prerot)
    """
    quat_fn = quat_to_rotmat_xyzw if quat_convention == "xyzw" else quat_to_rotmat_wxyz

    global_pos = np.zeros((127, 3))
    global_rot = np.zeros((127, 3, 3))
    global_scale = np.zeros((127, 1))

    for j in range(127):
        parent = int(joint_parents[j])

        local_trans = jd[j, :3]
        local_euler = jd[j, 3:6]
        local_scale_val = jd[j, 6:7]

        local_rot = euler_xyz_to_rotmat(local_euler)  # (3, 3)
        prerot_mat = quat_fn(joint_prerot[j])  # (3, 3)

        if prerot_order == "prerot_local":
            composed_rot = prerot_mat @ local_rot
        else:
            composed_rot = local_rot @ prerot_mat

        trans = joint_offsets[j] + local_trans
        scale = 1.0 + local_scale_val

        if parent == -1:
            global_pos[j] = trans
            global_rot[j] = composed_rot
            global_scale[j] = scale
        else:
            pr = global_rot[parent]
            pp = global_pos[parent]
            ps = global_scale[parent]

            global_pos[j] = pp + ps * (pr @ trans)
            global_rot[j] = pr @ composed_rot
            global_scale[j] = ps * scale

    return global_pos, global_rot, global_scale


# --- Also load MLX FK for comparison ---
import mlx.core as mx
from sam3d_mlx.mhr_body import MHRBodyModel
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

bm = MHRBodyModel(num_joints=127, num_verts=18439)
bm_weights = []
with safe_open("/tmp/sam3d-mlx-weights/model.safetensors", framework="numpy") as f:
    for key in f.keys():
        if key.startswith("mhr."):
            mhr_key = key[len("mhr."):]
            mapped = MHR_KEY_MAP.get(mhr_key)
            if mapped is not None:
                bm_weights.append((mapped, mx.array(f.get_tensor(key))))

bm.load_weights(bm_weights)
mx.eval(bm.parameters())

# Run MLX FK
jdofs_mx = mx.array(joint_dofs[np.newaxis, :].astype(np.float32))
mlx_skel, mlx_gpos, mlx_grot, mlx_gscale = bm._forward_kinematics(jdofs_mx)
mx.eval(mlx_skel, mlx_gpos, mlx_grot, mlx_gscale)
mlx_gpos_np = np.array(mlx_gpos[0])
mlx_grot_np = np.array(mlx_grot[0])

# --- Compare all convention combinations ---
print("\n=== Testing FK Convention Combinations ===")
print(f"Reference: PyTorch JIT skel_state positions")

conventions = [
    ("xyzw", "prerot_local"),
    ("xyzw", "local_prerot"),
    ("wxyz", "prerot_local"),
    ("wxyz", "local_prerot"),
]

for quat_conv, prerot_order in conventions:
    gpos, grot, gscale = run_fk(jd, joint_offsets, joint_prerot, joint_parents,
                                  quat_convention=quat_conv, prerot_order=prerot_order)

    pos_diff = np.abs(gpos - pt_skel_np[:, :3]).max()
    mean_diff = np.abs(gpos - pt_skel_np[:, :3]).mean()

    print(f"\n  quat={quat_conv}, prerot={prerot_order}:")
    print(f"    max pos diff vs JIT: {pos_diff:.6f}")
    print(f"    mean pos diff vs JIT: {mean_diff:.6f}")

    # Also compare against MLX
    mlx_diff = np.abs(gpos - mlx_gpos_np).max()
    print(f"    max pos diff vs MLX: {mlx_diff:.6f}")

# --- Joint-by-joint trace for the best matching convention ---
print("\n=== Joint-by-Joint FK Trace (first 20 joints) ===")
print(f"Convention: xyzw, prerot @ local (same as MLX implementation)")

gpos_xyzw, grot_xyzw, gscale_xyzw = run_fk(
    jd, joint_offsets, joint_prerot, joint_parents,
    quat_convention="xyzw", prerot_order="prerot_local"
)

print(f"\n{'Joint':>5} {'Parent':>6} {'MLX pos':>40} {'JIT pos':>40} {'Diff':>10}")
print("-" * 110)
for j in range(min(20, 127)):
    parent = int(joint_parents[j])
    mlx_p = mlx_gpos_np[j]
    jit_p = pt_skel_np[j, :3]
    diff = np.linalg.norm(mlx_p - jit_p)
    flag = " ***" if diff > 0.1 else ""
    print(f"  {j:>3}  {parent:>4}  [{mlx_p[0]:>10.4f},{mlx_p[1]:>10.4f},{mlx_p[2]:>10.4f}]"
          f"  [{jit_p[0]:>10.4f},{jit_p[1]:>10.4f},{jit_p[2]:>10.4f}]  {diff:>8.4f}{flag}")

# Find first divergent joint
print("\n=== First joint with significant FK divergence ===")
for j in range(127):
    diff = np.linalg.norm(mlx_gpos_np[j] - pt_skel_np[j, :3])
    if diff > 0.01:
        parent = int(joint_parents[j])
        print(f"Joint {j} (parent={parent}): position diff = {diff:.6f}")
        print(f"  MLX  pos: {mlx_gpos_np[j]}")
        print(f"  JIT  pos: {pt_skel_np[j, :3]}")

        # Check parent
        if parent >= 0:
            pdiff = np.linalg.norm(mlx_gpos_np[parent] - pt_skel_np[parent, :3])
            print(f"  Parent {parent} diff: {pdiff:.6f}")

        # Check local rotation
        euler = jd[j, 3:6]
        print(f"  Local euler: {euler}")
        print(f"  Local trans: {jd[j, :3]}")
        print(f"  Joint offset: {joint_offsets[j]}")
        print(f"  Prerotation quat: {joint_prerot[j]}")

        # Compare rotation matrices
        print(f"  MLX global_rot:\n{mlx_grot_np[j]}")

        # Compute numpy FK rotation for this joint
        local_rot = euler_xyz_to_rotmat(euler)
        prerot_mat = quat_to_rotmat_xyzw(joint_prerot[j])
        composed = prerot_mat @ local_rot
        print(f"  Numpy local_rot:\n{local_rot}")
        print(f"  Numpy prerot_mat:\n{prerot_mat}")
        print(f"  Numpy composed (prerot@local):\n{composed}")

        if parent >= 0:
            print(f"  MLX parent global_rot:\n{mlx_grot_np[parent]}")

        break  # Just show the first one

# --- Check if PyTorch uses a different euler convention ---
print("\n=== Checking if PyTorch JIT uses a different euler convention ===")
# Try running FK with different euler conventions to see which matches JIT

def euler_zyx_to_rotmat(angles):
    """ZYX euler -> rotation matrix. R = Rx @ Ry @ Rz"""
    z, y, x = angles[..., 0], angles[..., 1], angles[..., 2]
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    R = np.zeros(angles.shape[:-1] + (3, 3))
    R[..., 0, 0] = cy * cz
    R[..., 0, 1] = -cy * sz
    R[..., 0, 2] = sy
    R[..., 1, 0] = sx * sy * cz + cx * sz
    R[..., 1, 1] = -sx * sy * sz + cx * cz
    R[..., 1, 2] = -sx * cy
    R[..., 2, 0] = -cx * sy * cz + sx * sz
    R[..., 2, 1] = cx * sy * sz + sx * cz
    R[..., 2, 2] = cx * cy
    return R


def run_fk_euler_variant(jd, joint_offsets, joint_prerot, joint_parents,
                          euler_fn, quat_fn, prerot_order):
    """FK with arbitrary euler and quat functions."""
    global_pos = np.zeros((127, 3))
    global_rot = np.zeros((127, 3, 3))
    global_scale = np.zeros((127, 1))

    for j in range(127):
        parent = int(joint_parents[j])
        local_trans = jd[j, :3]
        local_euler = jd[j, 3:6]
        local_scale_val = jd[j, 6:7]

        local_rot = euler_fn(local_euler)
        prerot_mat = quat_fn(joint_prerot[j])

        if prerot_order == "prerot_local":
            composed_rot = prerot_mat @ local_rot
        else:
            composed_rot = local_rot @ prerot_mat

        trans = joint_offsets[j] + local_trans
        scale = 1.0 + local_scale_val

        if parent == -1:
            global_pos[j] = trans
            global_rot[j] = composed_rot
            global_scale[j] = scale
        else:
            pr = global_rot[parent]
            pp = global_pos[parent]
            ps = global_scale[parent]
            global_pos[j] = pp + ps * (pr @ trans)
            global_rot[j] = pr @ composed_rot
            global_scale[j] = ps * scale

    return global_pos


# Try all combinations
euler_fns = {
    "xyz": euler_xyz_to_rotmat,
    "zyx": euler_zyx_to_rotmat,
}
quat_fns = {
    "xyzw": quat_to_rotmat_xyzw,
    "wxyz": quat_to_rotmat_wxyz,
}
prerot_orders = ["prerot_local", "local_prerot"]

print(f"\n{'Euler':>5} {'Quat':>5} {'Prerot Order':>15} {'Max Pos Diff':>15} {'Mean Pos Diff':>15}")
print("-" * 60)
for euler_name, euler_fn in euler_fns.items():
    for quat_name, quat_fn in quat_fns.items():
        for prerot_order in prerot_orders:
            gpos = run_fk_euler_variant(
                jd, joint_offsets, joint_prerot, joint_parents,
                euler_fn, quat_fn, prerot_order
            )
            max_diff = np.abs(gpos - pt_skel_np[:, :3]).max()
            mean_diff = np.abs(gpos - pt_skel_np[:, :3]).mean()
            flag = " <<<" if max_diff < 1.0 else ""
            print(f"  {euler_name:>3}  {quat_name:>4}  {prerot_order:>13}  {max_diff:>13.6f}  {mean_diff:>13.6f}{flag}")
