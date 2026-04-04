# SAM 3D Body — Deep Dive Research

> Research date: 2026-04-03
> Purpose: Understand SAM 3D Body architecture for MLX porting and SamPlaysBaseball integration

## Model Identity

- **Name:** SAM 3D Body (3DB)
- **Published:** November 19, 2025 by Meta
- **Paper:** arXiv:2511.15586
- **Code:** github.com/facebookresearch/sam-3d-body
- **Weights:** facebook/sam-3d-body-dinov3 (HuggingFace, gated access)
- **License:** CC BY-NC 4.0 (non-commercial)
- **Two variants:** DINOv3-H+ (840M params, primary) and ViT-H (631M params, alternative)

## What It Does

Single-image full-body 3D human mesh recovery. Given one photo of a person, outputs:
- 18,439-vertex 3D mesh (LOD1)
- 127 skeleton joints with rotations
- 70 keypoints (body, hands, feet) in 3D and 2D
- Pose parameters (136 dims), shape parameters (45 dims)
- Camera translation

Uses MHR (Momentum Human Rig) instead of SMPL. MHR decouples skeleton from surface mesh — cleaner joint angle measurements.

## Architecture Overview

```
  INPUT: RGB image (any size)
    │
    ▼
  ┌─────────────────┐
  │ Human Detection  │  ViTDet or SAM3
  │ → bounding box   │
  └────────┬────────┘
           │ crop to 512x512
           ▼
  ┌─────────────────┐
  │ DINOv3-H+       │  840M params
  │ Backbone         │  patch_size=16, embed_dim=1280
  │ (ViT-H variant)  │  32 layers, 20 heads, SwiGLU FFN
  │                  │  RoPE + 4 register tokens
  └────────┬────────┘
           │ feature map: [B, 32, 32, 1280]
           ▼
  ┌─────────────────┐
  │ Camera Encoder   │  Fourier-encode pixel rays (99 channels)
  │                  │  Fuse with image features via Conv1x1
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ Prompt Encoder   │  Optional: keypoints [B, N, 3] or masks [B, 1, H, W]
  │ (optional)       │  70 learned keypoint embeddings
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ Transformer      │  5 layers, each with:
  │ Decoder          │  - Self-attention on ~145 tokens
  │                  │  - Cross-attention to image features
  │                  │  - FFN (SwiGLU)
  │                  │  - Intermediate prediction each layer
  │                  │  - Dynamic PE update from predicted joints
  └────────┬────────┘
           │ pose token → 519D parameter vector
           ▼
  ┌─────────────────┐
  │ MHR Head         │  FFN projects to 519 dims:
  │                  │  - 6D global rotation (6)
  │                  │  - Body pose continuous (260)
  │                  │  - Shape params (45)
  │                  │  - Scale params (28)
  │                  │  - Hand pose PCA (108)
  │                  │  - Face expression (72)
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ MHR Body Model   │  TorchScript JIT or native MHR package
  │ (Forward         │  Linear Blend Skinning + pose correctives
  │  Kinematics)     │  Input: shape(45) + params(204) + expr(72)
  └────────┬────────┘  Output: vertices(18439,3) + joints(127,8)
           │
           ▼
  ┌─────────────────┐
  │ Camera Head      │  Predicts (s, tx, ty) → compute tz from focal
  │ (Projection)     │  Full perspective projection
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ Hand Re-crop     │  Project wrist joints → hand bounding boxes
  │ + Hand Decoder   │  2 more backbone passes (left + right hand)
  │ + Merge          │  Separate decoder, same architecture
  └────────┬────────┘
           │
           ▼
  OUTPUT: vertices, joints, keypoints, pose params, mesh faces
```

## Input Format (Exact)

### Image
- RGB image, any resolution
- Preprocessing: detect humans → crop per person → affine transform to **512x512**
- Normalize: `(img - IMAGE_MEAN) / IMAGE_STD` (values from model config)
- ToTensor: float32 [0, 1]
- For ViT-H variant: width-cropped to 256x192 effective
- For DINOv3: full 512x512 square crop

### Bounding Box
- Format: `[x1, y1, x2, y2]` (xyxy)
- Padded with factor 1.25
- Aspect ratio fixed to 0.75 (w/h)

### Camera Intrinsics
- Default: `focal = sqrt(h^2 + w^2)`, principal point at image center
- Optional: MoGe2 FOV estimator for better accuracy
- Camera rays computed as meshgrid → undo affine → normalize by focal → `[B, 2, H, W]`

### Prompts (Optional)
- **Keypoints:** `[B, N, 3]` where dim 3 = (x_norm, y_norm, label)
  - label -2 = invalid, -1 = negative, 0-69 = joint index
- **Masks:** `[B, 1, H, W]` binary (white = person)
- Both are optional — model works in automatic mode

## Output Format (Exact)

Per detected person:

| Field | Shape | Description |
|-------|-------|-------------|
| pred_vertices | [18439, 3] | Mesh vertices in meters |
| pred_keypoints_3d | [70, 3] | 3D joint positions in meters |
| pred_keypoints_2d | [70, 2] | 2D projections in original image coords |
| pred_cam_t | [3] | Camera translation (tx, ty, tz) |
| focal_length | scalar | Estimated focal length |
| pred_joint_coords | [127, 3] | All skeleton joint coordinates |
| pred_global_rots | [127, 3, 3] | Joint global rotation matrices |
| body_pose_params | [133] | MHR body parameters |
| hand_pose_params | [108] | Hand PCA coefficients |
| shape_params | [45] | Identity shape |
| scale_params | [28] | Bone lengths |
| faces | [36874, 3] | Mesh face indices (constant) |

## MHR Body Model

### Skeleton
- 127 joints, each with 7 DoF: 3 translation, 3 rotation (Euler XYZ), 1 scale
- Transform chain: `Tw = Tp * Toff * Tt * Tprerot * Trot * Ts`
- Skeleton is independent of surface mesh (unlike SMPL)

### Parameters
- 204 model params: 136 pose + 68 skeleton transformation
- 45 shape params: 20 body PCA + 20 head PCA + 5 hand PCA
- 72 facial expression blendshapes (FACS-based)

### Mesh LODs

| LOD | Vertices |
|-----|----------|
| 0   | 73,639   |
| 1   | 18,439 (SAM 3D Body default) |
| 2   | 10,661   |
| 3   | 4,899    |
| 4   | 2,461    |
| 5   | 971      |

### Forward Pass
```
X(beta, theta) = LBS(X_tilde, Bk(beta_k), theta, omega)
X_tilde = X_bar + Bs(beta_s) + Bf(beta_f) + Bp(theta)
```
Where Bs = shape blendshapes, Bf = face blendshapes, Bp = pose correctives (two-stage MLPs)

### Python API
```python
mhr_model = MHR.from_files(device=torch.device("cpu"), lod=1)
vertices, skeleton_state = mhr_model(identity_coeffs, model_parameters, face_expr_coeffs)
# identity_coeffs: [batch, 45]
# model_parameters: [batch, 204]
# face_expr_coeffs: [batch, 72]
```

## DINOv3-H+ Backbone Details

| Property | Value |
|----------|-------|
| Parameters | 840M |
| Patch size | 16 |
| Embed dim | 1280 |
| Attention heads | 20 |
| Layers | 32 |
| FFN type | SwiGLU (the "+" in H+) |
| Position encoding | RoPE (rotary, not learned) |
| Register tokens | 4 |
| Training | Distilled from ViT-7B teacher |

Differences from DINOv2:
- SwiGLU FFN instead of standard MLP
- RoPE instead of learned positional embeddings
- Register tokens (4 learnable tokens)
- Trained at larger scale (7B teacher)

Code: `torch.hub.load("facebookresearch/dinov3", name, ...)`
Features: `backbone.get_intermediate_layers(x, n=1, reshape=True, norm=True)[-1]`
Output: `[B, C, H, W]` where C=1280, H=W=32 (for 512x512 input)

## Inference Pipeline (Full)

Total per person: **3 backbone passes** (body + 2 hands) + **2 decoder passes** (initial + refinement)

1. **Detect humans** — ViTDet or SAM3 produces bboxes
2. **FOV estimation** — MoGe2 predicts camera K (optional)
3. **Body crop** — Affine warp to 512x512, normalize
4. **Compute camera rays** — Meshgrid → undo affine → normalize by focal → `[B, 2, H, W]`
5. **Backbone pass #1** — DINOv3-H+ encodes body crop → `[B, 1280, 32, 32]`
6. **Camera ray conditioning** — Fourier rays (99ch) fused into features via Conv1x1 + LayerNorm
7. **CLIFF condition** — `(cx-W/2)/f, (cy-H/2)/f, bboxSize/f` → 3D vector
8. **Prepare tokens** — pose token (519D → linear → 1024D) + prompts + 70 kp2d + 70 kp3d + 2 hand = ~145 tokens
9. **Body decoder** — 5 transformer layers with intermediate MHR FK at each layer
10. **MHR forward** — Pose token → 519D → decompose → MHR body model → vertices + joints
11. **Camera projection** — Predict (s,tx,ty), compute tz, project 3D→2D
12. **Hand detection** — From hand tokens, predict hand bboxes
13. **Hand crop + encode** — Backbone pass #2 (left hand) + #3 (right hand), each 512x512
14. **Hand decode** — Separate decoder, same architecture
15. **Merge** — Compare wrist alignment, merge hand params if better
16. **Refinement pass** — Feed merged 2D keypoints as prompts, re-run body decoder

## Dependencies

```
Python 3.11
PyTorch (latest, CUDA required for original)
pytorch-lightning
detectron2 (specific commit a1ce2f9)
roma (rotation math — euler/rotmat/quat conversions)
einops
timm
hydra-core, hydra-colorlog
yacs, omegaconf
opencv-python
scikit-image
pyrender (visualization)
huggingface_hub
torchvision
networkx==3.2.1
Optional: flash_attn, MoGe, sam3
```

Key PyTorch ops: `F.scaled_dot_product_attention`, `torch.jit.load` (MHR), `roma.*`

## Model Weights

HuggingFace: `facebook/sam-3d-body-dinov3` (gated, requires approval)

Files:
- `model.ckpt` — PyTorch Lightning checkpoint
- `assets/mhr_model.pt` — MHR body model (TorchScript JIT)
- `model_config.yaml` — YACS config

MHR model loaded as either:
- Native: `from mhr.mhr import MHR; MHR.from_files(lod=1)`
- Fallback: `torch.jit.load(mhr_model_path)`

## MLX Porting: SAM 3 as Reference

SAM 3 (segmentation) port in mlx-vlm (15 files under mlx_vlm/models/sam3/):

### Key PyTorch → MLX Translations

| PyTorch | MLX |
|---------|-----|
| Conv2d (BCHW) | Conv2d (BHWC, channel-last) |
| ConvTranspose2d | ConvTranspose2d (channel-last) |
| F.scaled_dot_product_attention | mx.fast.scaled_dot_product_attention |
| nn.GELU | nn.gelu |
| F.interpolate | nn.Upsample |
| torch.pad | mx.pad |
| einsum (RoPE) | Manual compute_axial_cis + apply_rotary_enc |

### Weight Conversion Pattern
1. Conv2d weights: (out, in, H, W) → (out, H, W, in) via transpose(0, 2, 3, 1)
2. ConvTranspose2d: (in, out, H, W) → (out, H, W, in) via transpose(1, 2, 3, 0)
3. Fused QKV: split in_proj_weight into separate q_proj, k_proj, v_proj
4. Key renaming: regex-based mapping from Meta naming to HF naming
5. Save as safetensors

### MLX Optimizations Used
- Lazy evaluation with strategic mx.eval() to bound memory
- mx.fast.scaled_dot_product_attention (Metal kernel)
- wired_limit context manager for Metal memory management
- mx.clear_cache() during long sequences
- No quantization for vision models (bf16 only)

### SAM 3 Performance on Apple Silicon
- SAM3: 0.924 score, 1900ms, 4234 MB peak
- SAM3.1: 0.935 score, 1843ms, 6029 MB peak

## MLX Port Feasibility Assessment

### Difficulty by Component

| Component | Difficulty | Notes |
|-----------|-----------|-------|
| DINOv3-H+ backbone | HARD | RoPE ViT with SwiGLU. Similar to SAM3 ViT but different PE scheme. |
| Camera encoder | EASY | Conv1x1 + Fourier features. Pure math. |
| Prompt encoder | EASY | Embeddings + position encoding. |
| Transformer decoder | MEDIUM | Standard self-attn + cross-attn. Dynamic PE update is tricky. |
| MHR head (param regression) | EASY | Linear layers + reshape. |
| MHR body model | HARD | TorchScript JIT blob. Need to reverse-engineer or rewrite LBS + FK + pose correctives. |
| Rotation math (roma) | MEDIUM | 6D continuous → rotmat → euler. Port roma ops to MLX. |
| Weight conversion | MEDIUM | Conv transpose + key remapping + possible QKV splitting. |
| Hand re-crop pipeline | MEDIUM | Multi-pass coordination. |

### Key Challenges

1. **MHR body model** — This is a TorchScript JIT blob. You need to either:
   - Reverse-engineer the JIT model's operations (LBS, FK, blendshapes)
   - Rewrite using the MHR Python package (github.com/facebookresearch/MHR)
   - Port the MHR forward pass to MLX (matrix ops + skinning)

2. **DINOv3 backbone** — Loaded via torch.hub. Need to port the full ViT with RoPE and SwiGLU to MLX. SAM3's ViT port is a reference but uses different PE scheme.

3. **Multi-pass pipeline** — 3 backbone passes + 2 decoder passes per person. Need careful memory management on Apple Silicon.

4. **Intermediate FK at each decoder layer** — The decoder runs full forward kinematics at every layer for dynamic PE. This means MHR must be fast, not just correct.

### Memory Estimate

- DINOv3-H+ weights: 840M params × 2 bytes (bf16) = ~1.68 GB
- MHR model: ~50-100 MB (smaller)
- Decoder: ~100-200 MB
- Activations (512x512 input, 32x32 features): ~2-4 GB
- Total estimated: **4-6 GB peak** (fits in 16GB M-series Mac)

### Speed Estimate

SAM3 (860M params ViT + decoder) runs at ~1900ms on Apple Silicon.
SAM 3D Body (840M params DINOv3 + decoder + MHR) would likely be **2-4 seconds per crop**.
With 3 backbone passes per person: **6-12 seconds per person**.
Fast SAM 3D Body techniques could reduce this to **2-4 seconds** total.

## SMPL Compatibility

MHR-to-SMPL conversion:
- Iterative optimization: ~4-5 seconds/frame (slow)
- Fast SAM 3D Body's MLP replacement: 3-layer MLP (4500→512→256→76), runs at ~2000 FPS
- Output: global orientation (3) + body pose (63) + shape (10) = 76 SMPL params

## Fast SAM 3D Body (Acceleration)

arXiv:2603.15603 (March 2026)

Key optimizations:
- Parallelized multi-crop: batch all 3 crops into one backbone pass at 384x384
- Streamlined decoder: architecture-aware pruning
- Direct feedforward SMPL mapping: MLP replaces iterative fitting (10,000x faster)
- Result: **10.9x end-to-end speedup**, ~65ms/frame on RTX 5090
- Training-free (works with original SAM 3D Body weights)

## Sources

- SAM 3D Body repo: https://github.com/facebookresearch/sam-3d-body
- SAM 3D Body HuggingFace: https://huggingface.co/facebook/sam-3d-body-dinov3
- Meta blog: https://ai.meta.com/blog/sam-3d/
- Fast SAM 3D Body: https://github.com/yangtiming/Fast-SAM-3D-Body
- Fast SAM 3D Body paper: https://arxiv.org/abs/2603.15603
- MHR: https://github.com/facebookresearch/MHR
- DINOv3: https://github.com/facebookresearch/dinov3
- SAM-Body4D: https://github.com/gaomingqi/sam-body4d
- SAM4Dcap: https://arxiv.org/html/2602.13760
- mlx-vlm: https://github.com/Blaizzy/mlx-vlm
- fal.ai: https://fal.ai/models/fal-ai/sam-3/3d-body
