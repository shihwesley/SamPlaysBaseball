## Code Review
P0: 0 | P1: 0 | P2: 3 | P3: 4

### P2: PitcherMesh.tsx traversal runs on every render
`scene.traverse()` to check `hasMorphTargets` runs synchronously on every render cycle. Should be memoized with `useMemo` or computed once in a `useEffect`. Low-impact at small frame counts but accumulates with complex scenes.

### P2: TimelineScrubber useEffect missing stable onFrameChange reference
The `useEffect` for playback interval includes `onFrameChange` in deps but it's passed from parent and recreates on every render. Will cause interval to reset continuously if parent doesn't memoize the callback. Fix: wrap `onFrameChange` in `useCallback` at call sites, or use `useRef` for the callback inside TimelineScrubber.

### P2: GhostOverlay clones scene on every render
`clone(scene)` from SkeletonUtils is called inside the component body, meaning it runs on every render. Should be wrapped in `useMemo([scene])`.

### P3: comparison.py sets Buffer after building meshes
Buffer is appended after all meshes are built, but bufferViews reference buffer index 0. This works because GLTF2() starts with no buffers, but it's fragile. Index 0 is only valid because no other buffers exist.

### P3: ground_plane.py SVD may return non-deterministic normal direction
`np.linalg.svd` returns V^T rows in arbitrary sign — the ground normal could point up or down. The `np.cross(ground_normal, up)` and `c = np.dot(ground_normal, up)` computation handles this mathematically (rotation will flip if c < 0), but `abs(c + 1.0) < 1e-8` edge case check could miss near-antiparallel case. Low risk in practice.

### P3: frontend/src/lib/api.ts — no request timeout or retry
`apiFetch` has no timeout or AbortController. Long video uploads or slow analysis queries will hang indefinitely. Acceptable for MVP; production would add `AbortSignal.timeout(30000)`.

### P3: MoundScene.tsx camera position doesn't update when cameraPreset changes
The `canvas camera` prop is set once at mount time. Changing `cameraPreset` after mount won't move the camera. CameraPresets component calls `onPresetChange` but nothing in MoundScene re-positions the camera. Fix: use `useThree` to imperatively set camera, or pass preset to a separate Camera component inside Canvas.
