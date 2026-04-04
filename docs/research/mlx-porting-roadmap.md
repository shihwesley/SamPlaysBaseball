# SAM 3D Body → MLX Porting Roadmap

> Research date: 2026-04-03
> Status: Research complete. Ready for implementation.

## Architecture Comparison: SAM3 (ported) vs SAM 3D Body (to port)

| Component | SAM3 (MLX, done) | SAM 3D Body (PyTorch, to port) |
|-----------|-------------------|-------------------------------|
| Backbone | ViT-H (1024d, 32 layers) | DINOv3-H+ (1280d, 32 layers) |
| Patch size | 14 | 16 |
| Position encoding | Learned + 2D axial RoPE | RoPE (different impl) |
| FFN | Standard MLP (GELU) | SwiGLU |
| Attention | Windowed + global | Standard (no windowing) |
| Decoder | DETR (encoder+decoder) | Promptable decoder (5 layers) |
| Output | Segmentation masks | 3D body mesh + joints |
| Extra | CLIP text encoder, FPN neck | MHR body model, camera encoder |
| Model size | ~860M params | ~840M params |

## Three Porting Paths

### Path A: Full MLX Port (Recommended)
Port every component to MLX. Pure Apple Silicon, no PyTorch dependency.

**Pros:** Fastest inference, cleanest integration, best portfolio piece.
**Cons:** Most work. MHR body model is the hard part.
**Effort:** 3-4 weeks.

### Path B: Hybrid (MLX backbone + PyTorch MHR)
Port the backbone and decoder to MLX, keep MHR as PyTorch (JIT model on CPU).

**Pros:** Faster to ship. MHR is small (~50ms) vs backbone (~2s).
**Cons:** PyTorch dependency remains. Data transfer between MLX and PyTorch.
**Effort:** 2-3 weeks.

### Path C: PyTorch MPS (No MLX)
Run the original PyTorch model on MPS backend. No porting needed.

**Pros:** Works today (probably). Zero porting effort.
**Cons:** Slower (MPS is a translation layer). Not a portfolio contribution.
**Effort:** 1-2 days (testing + debugging MPS compatibility).

## Recommended: Path A (Full MLX Port)

### Phase 1: DINOv3-H+ Backbone (1 week)

**Reference:** mlx-vlm SAM3 `vision.py` (ViT backbone)

The DINOv3 backbone in SAM 3D Body is loaded via `torch.hub.load("facebookresearch/dinov3", ...)`. It's a standard ViT-H with:
- 32 transformer layers
- embed_dim=1280, num_heads=20, head_dim=64
- SwiGLU FFN (not standard MLP)
- RoPE position encoding
- 4 register tokens
- patch_size=16

**Porting steps:**
1. Create `config.py` with backbone config dataclass
2. Port `PatchEmbedding` — Conv2d(3, 1280, k=16, s=16) in BHWC format
3. Port `SwiGLUFFN` — Linear(D, 2*H) → split → silu(x1)*x2 → norm → Linear(H, D)
4. Port RoPE — different from SAM3's axial RoPE. DINOv3 uses standard 1D RoPE applied after reshaping patches
5. Port attention blocks — standard ViT attention (no windowing)
6. Port the full backbone with `get_intermediate_layers()` API

**Key differences from SAM3 port:**
- No windowed attention (simpler)
- SwiGLU instead of GELU MLP (small change)
- Different RoPE implementation
- No FPN neck needed
- 1280d instead of 1024d

### Phase 2: Camera + Prompt + Decoder (1 week)

**CameraEncoder** (`camera_embed.py`):
- Fourier position encoding of pixel rays (99 channels)
- Conv2d(D+99, D, 1) + LayerNorm2d
- Bilinear interpolation for rays: use `mx.image.resize` or manual impl
- Port: straightforward

**PromptEncoder** (`prompt_encoder.py`):
- PositionEmbeddingRandom — register_buffer with gaussian matrix, sin/cos encoding
- Per-joint nn.Embedding(1, D) × 70 joints
- Optional mask CNN (Conv2d chain)
- Port: straightforward

**PromptableDecoder** (`promptable_decoder.py`):
- 5 TransformerDecoderLayer blocks
- Each: self-attn → cross-attn → FFN
- Intermediate MHR forward kinematics at each layer (for dynamic PE)
- Callback-based token update
- Port: medium difficulty (dynamic PE is the tricky part)

**TransformerDecoderLayer** (`transformer.py`):
- MultiheadAttention: fused QKV → reshape → SDPA → project out
- FFN: Linear → ReLU → Dropout → Linear
- DropPath / LayerScale: identity / scalar multiply for inference
- Port: straightforward (drop Dropout/DropPath for inference)

### Phase 3: MHR Body Model (1 week) — THE HARD PART

The MHR model does 4 operations:
1. **Blend shape forward:** `coeffs @ shape_vectors + base_mesh` → rest pose vertices
2. **Parameter transform:** Linear map from 204 model params → joint parameters (per-joint 7-DOF)
3. **Forward kinematics:** Chain of matrix multiplications through 127-joint skeleton tree
4. **Linear blend skinning:** Weighted sum of joint-transformed vertex positions

**Two sub-paths for MHR:**

**Option 3A: Reverse-engineer JIT model**
The `mhr_model.pt` TorchScript file contains the complete forward pass as a frozen graph. You can inspect it:
```python
model = torch.jit.load("mhr_model.pt", map_location="cpu")
print(model.code)  # See the TorchScript source
print(model.graph)  # See the computation graph
```
Then reimplement each operation in MLX.

**Option 3B: Reimplement from MHR Python source**
The MHR Python package (`mhr/mhr.py`) shows the forward pass clearly:
```python
# 1. Blend shapes
rest_pose = character_torch.blend_shape.forward(coeffs)
# 2. Parameter transform + FK
joint_parameters = character_torch.model_parameters_to_joint_parameters(params)
skel_state = character_torch.joint_parameters_to_skeleton_state(joint_parameters)
# 3. Pose correctives (MLP)
corrective_offsets = pose_correctives_model(joint_parameters)
rest_pose += corrective_offsets
# 4. Skinning
verts = character_torch.skin_points(skel_state, rest_pose)
```

These operations use `pymomentum.torch.character` (C++ bindings), but the math is standard:
- Blend shapes: matrix multiply + add
- Parameter transform: linear projection
- FK: recursive joint chain traversal (parent-to-child matrix multiply)
- LBS: `v = sum_j(w_j * T_j @ v_rest)` where w_j are skinning weights, T_j are joint transforms

All of this is pure linear algebra. No exotic ops.

**Pose correctives model** (`MHRPoseCorrectivesModel`):
- Pure PyTorch `nn.Sequential` (MLPs)
- Takes joint parameters → 6D rotation features → MLP → vertex offsets
- Direct port to MLX

**Data files needed (from MHR assets):**
- `compact_v6_1.model` — model parameterization (joint tree, parameter transform)
- `corrective_activation.npz` — sparse activation masks for corrective MLPs
- `corrective_blendshapes_lod1.npz` — pose corrective blendshape weights
- `lod1.fbx` — rig definition (joint hierarchy, skinning weights, base mesh)
- `mhr_model.pt` — TorchScript fallback model

### Phase 4: Camera Head + Rotation Utils + Integration (0.5 week)

**PerspectiveHead** (`camera_head.py`):
- FFN: Linear(D) → 3 values (s, tx, ty)
- Perspective projection math
- Port: trivial

**Rotation utilities** (`mhr_utils.py` + roma):
- `rot6d_to_rotmat` — Gram-Schmidt orthogonalization. ~15 lines of linear algebra
- `compact_cont_to_model_params_body` — 260D continuous → 133D Euler. atan2 + reshape
- `roma.rotmat_to_euler("ZYX", R)` — ~30 lines of atan2 math
- `roma.euler_to_rotmat("xyz", angles)` — ~20 lines of sin/cos/matmul
- `roma.unitquat_to_rotmat(q)` — ~15 lines of quaternion algebra
- Port: medium (careful about conventions, but pure math)

**Integration** (`sam3d_body.py`):
- Wire all components together
- Multi-pass pipeline (body → hands → merge → refine)
- Write `sanitize()` for weight conversion
- Write `convert_weights.py` for checkpoint conversion
- Port: medium (lots of plumbing)

### Phase 5: Weight Conversion + Testing (0.5 week)

**Weight conversion pipeline:**
1. Load `model.ckpt` (PyTorch Lightning format)
2. Extract `state_dict`
3. Remap keys (Meta naming → MLX naming)
4. Split fused QKV if present
5. Transpose Conv2d weights: (out,in,H,W) → (out,H,W,in)
6. Save as safetensors

**Testing:**
1. Run PyTorch model on test image → save outputs
2. Run MLX model on same image → compare outputs
3. Verify: vertices within 1e-4, keypoints within 1e-3
4. Benchmark: latency, peak memory

## Key Dimensions Reference

| Constant | Value |
|----------|-------|
| Backbone embed_dim | 1280 |
| Backbone layers | 32 |
| Backbone heads | 20 |
| Backbone patch_size | 16 |
| Input crop size | 512×512 |
| Feature map | 32×32×1280 |
| Decoder layers | 5 |
| Decoder dim | 1024 |
| MHR joints | 127 |
| MHR keypoints | 70 (body subset of 308) |
| MHR vertices (LOD1) | 18,439 |
| MHR faces | 36,874 |
| Pose params | 136 (body) |
| Shape params | 45 (20 body + 20 head + 5 hand) |
| Scale params | 28 |
| Hand params | 54 per hand (PCA) |
| Face params | 72 |
| Total regression | 519 dims |
| Backbone passes per person | 3 (body + 2 hands) |
| Decoder passes per person | 2 (initial + refinement) |

## PyTorch → MLX Translation Table

| PyTorch | MLX | Notes |
|---------|-----|-------|
| `nn.Linear` | `nn.Linear` | Direct |
| `nn.Conv2d(NCHW)` | `nn.Conv2d(NHWC)` | Transpose weights |
| `nn.Embedding` | `nn.Embedding` | Direct |
| `nn.LayerNorm` | `nn.LayerNorm` | Direct |
| `F.silu` | `nn.silu` | Direct |
| `F.gelu` | `nn.gelu` | Direct |
| `F.scaled_dot_product_attention` | `mx.fast.scaled_dot_product_attention` | No dropout_p |
| `F.interpolate(bilinear)` | Manual or `mx.image.resize` | No antialias |
| `F.normalize` | `x / mx.linalg.norm(x, keepdims=True)` | Manual |
| `torch.einsum` | `mx.einsum` | Direct |
| `torch.linalg.cross` | `mx.linalg.cross` | Direct |
| `torch.cat` | `mx.concatenate` | Direct |
| `torch.chunk` | `mx.split` | Direct |
| `torch.atan2` | `mx.arctan2` | Direct |
| `roma.*` | Reimplement (~100 lines total) | Pure math |
| `torch.jit.load` | Reimplement MHR forward | Hardest part |
| `register_buffer` | `self.x = mx.array(...)` | Non-trainable |
| `nn.Dropout` | Remove (inference only) | |
| `DropPath` | Remove (inference only) | |
| `.permute()` | `mx.transpose` | |
| `.contiguous()` | No-op | |

## Estimated Timeline

| Phase | Duration | Dependencies |
|-------|----------|-------------|
| 1. DINOv3-H+ backbone | 5-7 days | HuggingFace model access |
| 2. Camera + Prompt + Decoder | 5-7 days | Phase 1 |
| 3. MHR body model | 5-7 days | MHR assets download |
| 4. Rotation utils + Integration | 3-4 days | Phases 1-3 |
| 5. Weight conversion + Testing | 3-4 days | All phases |
| **Total** | **3-4 weeks** | |

## Estimated Memory

| Component | bf16 Size |
|-----------|----------|
| DINOv3-H+ weights | 1.68 GB |
| Decoder weights | ~200 MB |
| MHR model data | ~100 MB |
| Activations (512×512 input) | ~2-4 GB |
| **Total peak** | **~4-6 GB** |

Fits on any M-series Mac with 16 GB+ unified memory.

## Sources

- SAM 3D Body source: /tmp/sam-3d-body/
- mlx-vlm SAM3 source: /tmp/mlx-vlm/mlx_vlm/models/sam3/
- MHR source: /tmp/MHR/mhr/
- MHR paper: https://arxiv.org/abs/2511.15586
- DINOv3: https://github.com/facebookresearch/dinov3
- Meta Momentum: https://github.com/facebookresearch/momentum
