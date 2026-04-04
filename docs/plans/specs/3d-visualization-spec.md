---
name: 3d-visualization
phase: 3
sprint: 2
parent: api-layer
depends_on: [api-layer, mesh-export]
status: draft
created: 2026-02-16
updated: 2026-04-03
---

# 3D Visualization Spec

Three.js-based 3D mound scene with mesh replay and four overlay comparison modes. The showcase component.

## Requirements

- Render pitcher mesh on a 3D baseball mound in the browser
- Load GLB files from mesh-export pipeline (morph target animation)
- Skeleton overlay with MHR70 joint hierarchy
- Color-code joints by deviation from baseline
- Frame scrubber synced to delivery phases
- Four overlay comparison modes (see below)
- Camera orbit controls with preset angles

## Mound Scene

The pitcher throws from a regulation MLB mound in a 3D environment:
- Mound geometry: 18ft diameter, 10.5in rise, rubber at center
- Dirt/grass material distinction
- Home plate direction marker (visual reference for orientation)
- Optional: minimal backstop/batter silhouette for spatial context
- Ground-plane-aligned meshes from mesh-export placed on the rubber

## Four Overlay Comparison Modes

### 1. Ghost Overlay
- Pitch A rendered solid, Pitch B at 30% opacity, same position
- Both phase-synced (aligned to foot plant, MER, or release)
- See where the transparent body diverges from the solid one
- Use case: 1st inning fastball vs 7th inning fastball (fatigue detection)

### 2. Difference Highlighting
- Single mesh rendered, joints color-coded by deviation magnitude
- Green (normal) → yellow (1.0-1.5 SD) → orange (1.5-2.0 SD) → red (> 2.0 SD)
- Red pulse animation on joints exceeding threshold
- Use case: quick visual of which body parts deviate from baseline

### 3. Split-Sync Side-by-Side
- Two mound scenes rendered side by side
- Both orbit cameras linked (rotate one, both rotate)
- Both timelines phase-synced (same delivery phase shown simultaneously)
- Use case: fastball delivery vs changeup delivery (tipping detection)

### 4. Temporal Stroboscope
- Multiple deliveries of the same pitch type superimposed as semi-transparent layers
- Like a long-exposure photo of the delivery in 3D
- Shows consistency/scatter of the delivery path
- Use case: release point consistency, arm path repeatability

## Acceptance Criteria

- [ ] Mound scene: regulation mound geometry, ground-aligned mesh standing on rubber
- [ ] GLB loading: loads morph-target-animated GLBs from mesh-export
- [ ] Smooth playback at original FPS with morph target interpolation
- [ ] Skeleton overlay: MHR70 joints (70 points) + bone lines using official skeleton_info
- [ ] Deviation coloring: per-joint green → red based on Z-score from baseline
- [ ] Scrubber: play/pause, frame step, speed (0.25x-2x), phase markers on timeline
- [ ] Ghost overlay: two meshes, solid + transparent, phase-aligned
- [ ] Split-sync: dual viewports with linked orbit controls and synced timelines
- [ ] Temporal stroboscope: N meshes (5-20 pitches) stacked at 10-15% opacity each
- [ ] Difference highlighting: per-joint color, red pulse animation on outliers
- [ ] Camera presets: front (catcher view), side (1B/3B), overhead, behind pitcher
- [ ] Responsive layout, works at different viewport sizes
- [ ] Export: screenshot current view as PNG

## Technical Approach

Three.js with React Three Fiber (R3F). GLB loading via `useGLTF` hook — morph targets are native to Three.js `Mesh.morphTargetInfluences`. Mound geometry either authored in Blender and exported as GLB, or generated procedurally from MLB dimensions.

Ghost overlay: load comparison GLB from mesh-export (contains both meshes with different materials). Three.js renders both — the transparent mesh uses `MeshStandardMaterial` with `transparent: true, opacity: 0.3`.

Split-sync: two `<Canvas>` components with shared state for camera rotation and timeline position. React context syncs the controls.

Temporal stroboscope: load N pitch GLBs, freeze each at the same phase frame, render all in one scene at low opacity. Three.js handles alpha blending.

Deviation coloring: `InstancedMesh` for 70 joint spheres, per-instance color attribute updated each frame from pre-computed deviation Z-scores via baseline-comparison module.

## Files

| File | Purpose |
|------|---------|
| frontend/src/components/three/MoundScene.tsx | 3D mound environment + lighting |
| frontend/src/components/three/PitcherMesh.tsx | GLB mesh loader + morph target playback |
| frontend/src/components/three/SkeletonOverlay.tsx | MHR70 joint spheres + bone lines |
| frontend/src/components/three/DeviationColoring.tsx | Per-joint Z-score color mapping |
| frontend/src/components/three/GhostOverlay.tsx | Transparent comparison mesh |
| frontend/src/components/three/SplitSync.tsx | Dual viewport with linked controls |
| frontend/src/components/three/Stroboscope.tsx | Multi-pitch temporal overlay |
| frontend/src/components/three/TimelineScrubber.tsx | Phase-aware playback controls |
| frontend/src/components/three/CameraPresets.tsx | Preset camera angles |
| frontend/src/lib/mesh-loader.ts | Fetch + decode GLB mesh data |

## Tasks

1. Build mound scene (regulation geometry, materials, lighting, ground plane)
2. Implement GLB mesh loader with morph target animation playback
3. Build MHR70 skeleton overlay (joints + bones from official skeleton_info)
4. Build timeline scrubber with phase markers (foot plant, MER, release)
5. Implement ghost overlay mode (solid + transparent, phase-synced)
6. Implement split-sync mode (dual viewport, linked orbit + timeline)
7. Implement temporal stroboscope mode (N pitches stacked)
8. Add deviation color-coding per joint from baseline Z-scores
9. Add camera presets and screenshot export
10. Responsive layout and viewport management

## Dependencies

- Upstream: api-layer (provides analysis data), mesh-export (provides GLB files)
- Downstream: demo-mode, blender-render
