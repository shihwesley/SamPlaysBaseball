---
name: data-model
phase: 1
sprint: 1
parent: root
depends_on: []
status: draft
created: 2026-02-16
---

# Data Model Spec

The foundation layer. Every other spec depends on these structures.

## Requirements

- Define core data structures for pitch data, pitcher baselines, and analysis results
- Storage layer that works for both live processing and pre-computed demo mode
- Serialization for passing data between pipeline stages and to the API layer

## Acceptance Criteria

- [ ] PitchData model holds full SAM 3D Body output per pitch: `(T, 127, 3)` joints, `(T, 136)` pose params, `(45,)` shape params, metadata
- [ ] PitcherBaseline model stores per-pitcher, per-pitch-type statistical baselines
- [ ] AnalysisResult models for each of the 6 analysis modules
- [ ] Storage layer reads/writes to SQLite (structured queries) + Parquet (numpy arrays)
- [ ] All models serialize to JSON for API transport
- [ ] Sample data fixtures for testing downstream specs

## Technical Approach

SQLite for relational data (pitcher info, pitch metadata, analysis results). Parquet files for large numpy arrays (joint positions, pose parameters) — keeps the database lean while supporting efficient columnar reads. Pydantic models for validation + serialization.

## Files

| File | Purpose |
|------|---------|
| backend/app/models/pitch.py | PitchData, PitchMetadata |
| backend/app/models/baseline.py | PitcherBaseline |
| backend/app/models/analysis.py | AnalysisResult variants per module |
| backend/app/models/storage.py | SQLite + Parquet read/write |
| backend/app/models/__init__.py | Re-exports |
| backend/tests/test_models.py | Model validation tests |

## Tasks

1. Define PitchData and PitchMetadata models
2. Define PitcherBaseline model with statistical aggregation
3. Define AnalysisResult models for all 6 analysis types
4. Implement SQLite + Parquet storage layer
5. Create sample data fixtures

## Dependencies

- Upstream: none (root spec)
- Downstream: video-pipeline, sam3d-inference, feature-extraction, api-layer all consume these models
