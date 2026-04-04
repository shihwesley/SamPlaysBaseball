---
name: mlx-validation
phase: 3
sprint: 1
parent: mlx-port-manifest
depends_on: [mlx-inference]
status: draft
created: 2026-04-03
---

# MLX Validation — Numerical Parity + Performance

Validate MLX port produces identical results to PyTorch and measure speedup.

## Requirements

- Per-layer numerical comparison (backbone, decoder, mhr) vs PyTorch
- End-to-end vertex/joint comparison on real images
- Performance benchmarks: latency, throughput, memory
- TurboQuant accuracy impact measurement
- Regression test suite that runs in CI

## Acceptance Criteria

- [ ] Backbone output max diff < 1e-3 per element
- [ ] Decoder pose params max diff < 1e-3
- [ ] Vertex positions max diff < 1e-3 (18,439 vertices)
- [ ] Joint coordinates max diff < 1e-3 (127 joints)
- [ ] TurboQuant 3.5-bit: vertex diff < 5e-3 vs full precision
- [ ] MLX body-mode latency < 500ms on M2 Pro (vs 1.1s MPS PyTorch)
- [ ] MLX peak memory < 6GB (vs ~8GB PyTorch)

## Benchmark Matrix

| Metric | PyTorch CPU | PyTorch MPS | MLX | MLX + TQ 3.5 |
|--------|-------------|-------------|-----|--------------|
| Body mode latency | 11.5s | 1.1s | target <0.5s | target <0.5s |
| Full mode latency | ~35s | 4.8s | target <2s | target <2s |
| Peak memory | ~10GB | ~8GB | target <6GB | target <5GB |
| Vertex error vs PT | 0 | 0 | <1e-3 | <5e-3 |

## Files

| File | Action |
|------|--------|
| `tests/test_parity.py` | create (per-layer comparison) |
| `tests/test_e2e.py` | create (full pipeline comparison) |
| `tests/test_turboquant.py` | create (TQ accuracy impact) |
| `benchmarks/bench_inference.py` | create (latency/memory) |

## Tasks

1. Build per-layer comparison harness (run both PyTorch + MLX, diff outputs)
2. Run backbone parity test on 10 random images
3. Run full pipeline parity test on Ohtani pitch frames
4. Measure TurboQuant accuracy degradation at 2/3/3.5/4 bits
5. Run latency + memory benchmarks, produce comparison table
