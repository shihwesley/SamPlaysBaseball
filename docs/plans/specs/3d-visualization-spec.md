---
name: 3d-visualization
phase: 3
sprint: 2
parent: api-layer
depends_on: [api-layer]
status: draft
created: 2026-02-16
---

# 3D Visualization Spec

Three.js-based 3D mesh replay and analysis overlays in the browser. This is the showcase component — the thing that makes eyes go wide in a demo.

## Requirements

- Render SAM 3D Body mesh frame-by-frame in the browser
- Skeleton overlay with 127-joint hierarchy
- Color-code joints by deviation from baseline
- Frame scrubber (play/pause/step through delivery)
- Side-by-side comparison of two deliveries
- Ghost overlay: semi-transparent reference delivery behind current
- Camera orbit controls (rotate, zoom, pan the 3D view)

## Acceptance Criteria

- [ ] Mesh rendering: smooth playback of reconstructed body mesh at original FPS
- [ ] Skeleton overlay: lines connecting joints, joint spheres at positions
- [ ] Deviation coloring: green (normal) → yellow → red (high deviation) per joint
- [ ] Scrubber: timeline control, play/pause, frame step, speed control (0.25x to 2x)
- [ ] Phase markers on timeline: foot plant, MER, release marked visually
- [ ] Side-by-side mode: two synced 3D views (e.g., fastball vs changeup from same pitcher)
- [ ] Ghost overlay: transparent baseline delivery behind current delivery
- [ ] Difference highlighting: red pulse on joints that deviate from ghost
- [ ] Camera controls: OrbitControls, preset views (front, side, overhead, behind home plate)
- [ ] Responsive: works at different viewport sizes
- [ ] Export: screenshot current frame as PNG

## Technical Approach

Three.js with React Three Fiber (R3F) for React integration. BufferGeometry for mesh rendering (efficient for per-frame updates). InstancedMesh for joint spheres (127 instances, one draw call). Line segments for skeleton.

Mesh data transfer: fetch vertex data per pitch from API (compressed), decode client-side. For smooth playback, prefetch all frames of a pitch and step through them via requestAnimationFrame.

Ghost overlay: second mesh with MeshBasicMaterial at alpha 0.3, same animation sync. Deviation coloring: per-joint color attribute on InstancedMesh, updated each frame from pre-computed deviation data.

Reference tools from analytics-architecture.md: Open3D/PyVista for server-side preview, Rerun for development, Three.js/R3F for production web UI.

## Files

| File | Purpose |
|------|---------|
| frontend/src/components/three/MeshViewer.tsx | Main 3D mesh renderer |
| frontend/src/components/three/SkeletonOverlay.tsx | Joint spheres + bone lines |
| frontend/src/components/three/DeviationColoring.tsx | Per-joint color mapping |
| frontend/src/components/three/GhostOverlay.tsx | Transparent reference mesh |
| frontend/src/components/three/TimelineScrubber.tsx | Play/pause/step controls |
| frontend/src/components/three/ComparisonView.tsx | Side-by-side 3D views |
| frontend/src/components/three/CameraPresets.tsx | Preset camera angles |
| frontend/src/lib/mesh-loader.ts | Fetch + decode mesh data from API |

## Tasks

1. Set up React Three Fiber with OrbitControls and lighting
2. Implement mesh renderer with per-frame BufferGeometry updates
3. Build skeleton overlay (joints + bones) with instanced rendering
4. Add deviation color-coding per joint
5. Build timeline scrubber with phase markers
6. Implement side-by-side comparison mode
7. Build ghost overlay with difference highlighting
8. Add camera presets and screenshot export

## Dependencies

- Upstream: api-layer (provides mesh + joint + deviation data)
- Downstream: demo-mode
