## Performance Review
P0: 0 | P1: 0 | P2: 2 | P3: 3

### P2: Stroboscope.tsx loads N GLBs simultaneously
`glbUrls.map(url => PitcherMesh with url)` causes N concurrent useGLTF fetches. With 20 pitches this spawns 20 fetch + parse operations at once. For small N (<10) this is fine. For N=20 add a loading queue or progressive rendering. Deferred — N is bounded by UX (the slider caps at 20).

### P2: plotly.js adds ~3.5MB to bundle
`plotly.js` is dynamically imported via `next/dynamic` with `ssr: false`, which code-splits it correctly. The chunk will only load when a chart component is rendered. Already mitigated by dynamic import — no action needed for P2, just a size note.

### P3: SplitSync renders two Canvas elements
Each R3F Canvas creates its own WebGL context. Most browsers limit contexts to 8-16. Two contexts for SplitSync is fine, but combining with a MoundScene would be 3+. Acceptable for current use. Long-term: share a single context with multiple viewports.

### P3: GLB export for 18439 verts × N frames creates large blob
A 60-frame GLB with 18439 vertices = 60 × 18439 × 12 bytes ≈ 13MB uncompressed. The spec notes meshopt quantization via gltfpack as a post-process step — this is not currently applied in the exporter. The 5MB target requires gltfpack post-processing. Deferred — acceptable for dev, document requirement.

### P3: PitcherMesh.tsx loops morphTargetInfluences on every currentFrame change
Setting all morph target influences to 0 then setting one to 1 is O(N) per frame change. For 60 frames this is negligible. For 300+ frames (slow-motion) it would still be fast — no concern.
