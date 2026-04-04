---
name: blender-render
phase: 3
sprint: 2
parent: mesh-export
depends_on: [mesh-export, 3d-visualization]
status: draft
created: 2026-04-03
---

# Blender Render Spec

High-quality rendered video output of pitcher mechanics via Blender scripting. Produces polished comparison videos for reports, presentations, and social media.

## Requirements

- Automated Blender pipeline: import GLB, set up scene, render video
- Mound scene matching the Three.js viewer (consistent look across web and rendered output)
- Scripted camera moves: orbit, track arm, follow ball path
- Comparison renders: ghost overlay, side-by-side, stroboscope
- Batch rendering for multi-pitch sets

## Acceptance Criteria

- [ ] Blender Python script imports GLB mesh sequences and plays morph target animation
- [ ] Mound scene: regulation mound, dirt/grass materials, stadium-style lighting
- [ ] Camera presets: slow orbit (360 degrees over pitch duration), side view tracking, overhead
- [ ] Ghost overlay render: solid body + transparent body composited in Blender
- [ ] Side-by-side render: two mound scenes tiled horizontally
- [ ] Stroboscope render: multiple frozen poses composited at low opacity
- [ ] Output: MP4 at 1080p/4K, configurable FPS
- [ ] Batch mode: render N pitches with same camera setup in one script invocation
- [ ] Headless rendering: runs via `blender --background --python script.py`
- [ ] Annotation overlays: pitch type label, phase markers, deviation callouts burned into render

## Technical Approach

Blender Python API (`bpy`) for scene setup, material assignment, camera animation, and rendering. Import GLB via `bpy.ops.import_scene.gltf`. Morph target animation is preserved in Blender as shape keys — drive with keyframes.

Camera animation: keyframe camera position on a circular path around the mound for orbit shots. For tracking shots, parent camera to an empty that follows the throwing arm joint.

Materials: PBR skin-like material for the body mesh (subtle subsurface scattering). Mound uses a procedural dirt/clay texture. Grass plane with simple noise-based color variation.

Ghost overlay: duplicate mesh, assign glass/transparent shader at 0.3 alpha. Both animated in sync.

Stroboscope: import N copies of the mesh, each frozen at the same phase frame. Assign material with increasing transparency (first pitch most opaque).

Headless: script accepts CLI args for input GLB path, output video path, camera preset, and render quality. Runs unattended for batch processing.

## Files

| File | Purpose |
|------|---------|
| scripts/blender/render_pitch.py | Main Blender render script (single pitch) |
| scripts/blender/render_comparison.py | Ghost overlay + side-by-side renders |
| scripts/blender/render_stroboscope.py | Multi-pitch stroboscope render |
| scripts/blender/scene_setup.py | Mound geometry, materials, lighting |
| scripts/blender/camera_presets.py | Camera animation paths |
| scripts/blender/batch_render.py | Batch render wrapper |

## Tasks

1. Build mound scene setup script (geometry, materials, lighting)
2. Implement GLB import and morph target animation playback
3. Build camera preset animations (orbit, side track, overhead)
4. Implement ghost overlay render (solid + transparent mesh)
5. Implement side-by-side render
6. Implement stroboscope render
7. Add annotation overlays (text burns, phase markers)
8. Build batch render wrapper with CLI interface
9. Test headless rendering on macOS

## Dependencies

- Upstream: mesh-export (GLB files), 3d-visualization (scene design consistency)
- Downstream: ai-scouting-reports (rendered videos embedded in reports)
