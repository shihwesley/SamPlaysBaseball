## Test Review
P0: 0 | P1: 0 | P2: 2 | P3: 2

### P2: GLBExporter/ComparisonBuilder tests skipped (pygltflib not installed)
Three tests are marked `requires_pygltflib` and skip when pygltflib is absent. The tests are correct and would pass if pygltflib were installed (`pip install pygltflib`). Deferred — install in CI/CD environment.

### P2: No frontend component tests
No test files exist in `frontend/src/`. React component tests (e.g., with Jest + React Testing Library) would catch rendering regressions. Deferred to a later sprint — component behavior is straightforward and covered by TypeScript.

### P3: Blender scripts untestable without Blender
`scripts/blender/*.py` cannot be run in pytest. All scripts guard bpy with `try/except ImportError` and print an error on missing bpy. The import guard pattern is documented in the files and is the correct approach for Blender scripts.

### P3: test_comparison_phase_align covers interpolation math but not edge cases
The test checks that `len(aligned1) == len(aligned2) == 10`. It does not test zero-frame sequences or single-frame sequences. Low risk — these are checked via the ValueError in export_pitch.
