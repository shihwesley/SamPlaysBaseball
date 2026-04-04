---
name: mesh-export
phase: 2
sprint: 1
parent: sam3d-inference
depends_on: [sam3d-inference, feature-extraction]
status: draft
created: 2026-04-03
---

# Mesh Export Spec

Export SAM 3D Body mesh sequences to GLB/glTF for Three.js rendering and Blender import. Handles ground-plane alignment and mound placement.

## Requirements

- Export per-pitch mesh sequences as animated GLB files
- Ground-plane alignment: transform body from camera space to world space (feet on ground)
- Mound placement: position the aligned body on a pitcher's mound coordinate system
- Key-frame export: optionally export only phase-aligned frames (foot plant, MER, release)
- Batch export for multi-pitch comparison sets

## Acceptance Criteria

- [ ] Single-pitch GLB: animated mesh of one delivery, ground-aligned, playable in Three.js and Blender
- [ ] Ground-plane detection: ankle positions used to compute ground plane, body rotated/translated so feet sit on y=0
- [ ] Mound coordinate system: rubber at origin, home plate direction along -Z, mound slope modeled
- [ ] Key-frame mode: export only foot-plant/MER/release frames as separate meshes in one GLB (for comparison overlays)
- [ ] Comparison GLB: two pitches in one file, different materials (solid vs transparent), phase-aligned
- [ ] Skeleton export: joint positions as empties/bones in GLB for Three.js SkeletonHelper
- [ ] File size: single pitch GLB < 5MB (vertex quantization via meshopt)
- [ ] Metadata embedded: pitch type, pitcher ID, frame timestamps, phase markers as GLB extras

## Technical Approach

Use trimesh for mesh construction and GLB export. Ground-plane alignment:
1. For each frame, take ankle keypoints (MHR70 indices 13, 14) as ground contacts
2. Compute ground normal from ankle positions across frames (PCA on ankle trajectory gives ground plane)
3. Build rotation matrix to align ground normal with world Y-up
4. Translate so lowest ankle sits at mound rubber height

Animation: GLB morph targets (shape keys). All frames share the same face topology (36874 faces). Each frame's 18439 vertex positions become a morph target. Three.js animates between them via `morphTargetInfluences`.

For comparison GLBs: two meshes in the same scene, phase-aligned by feature-extraction phase boundaries. Second mesh gets a transparent material (`alphaMode: BLEND`, alpha: 0.3).

Mound geometry: hardcoded from MLB regulation dimensions (18ft diameter, 10.5in above baseline, 6in slope over 6ft toward plate). Export as a separate mesh node in the GLB.

## Files

| File | Purpose |
|------|---------|
| backend/app/export/glb.py | GLBExporter class, mesh sequence to GLB |
| backend/app/export/ground_plane.py | Ground-plane alignment from ankle positions |
| backend/app/export/mound.py | Mound geometry generation (MLB regulation) |
| backend/app/export/comparison.py | Multi-pitch comparison GLB builder |
| backend/tests/test_glb_export.py | Export tests, validates GLB loads in trimesh |

## Tasks

1. Implement ground-plane alignment from ankle keypoints
2. Build single-pitch GLB exporter with morph target animation
3. Generate MLB regulation mound geometry
4. Build comparison GLB exporter (two meshes, phase-aligned, different materials)
5. Add skeleton joint export as GLB nodes
6. Implement meshopt vertex quantization for file size
7. Test loading in Three.js and Blender

## Dependencies

- Upstream: sam3d-inference (mesh data), feature-extraction (phase boundaries for alignment)
- Downstream: 3d-visualization (loads GLBs), blender-render
