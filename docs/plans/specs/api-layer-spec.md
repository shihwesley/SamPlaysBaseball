---
name: api-layer
phase: 3
sprint: 1
parent: data-model
depends_on: [data-model, baseline-comparison, tipping-detection, fatigue-tracking, command-analysis, arm-slot-drift, timing-analysis]
status: draft
created: 2026-02-16
---

# API Layer Spec

FastAPI backend that connects the Python pipeline/analysis to the Next.js frontend.

## Requirements

- Video upload and processing trigger
- Pitch data retrieval (single pitch, filtered list)
- Analysis results for all 6 modules
- Pitcher profile endpoints (baseline, history, comparisons)
- Demo data endpoints (serve pre-computed results without GPU)
- WebSocket for real-time processing status updates

## Acceptance Criteria

- [ ] POST /api/upload — accept video + metadata, return job ID
- [ ] GET /api/jobs/{id} — processing status (queued, processing, complete, failed)
- [ ] WebSocket /api/ws/jobs/{id} — real-time progress updates during processing
- [ ] GET /api/pitchers — list all pitchers with summary stats
- [ ] GET /api/pitchers/{id} — pitcher profile with baseline data
- [ ] GET /api/pitchers/{id}/pitches — paginated pitch list with filters (date, type, outing)
- [ ] GET /api/pitches/{id} — single pitch with full joint data + features
- [ ] GET /api/pitches/{id}/mesh — 3D mesh data for Three.js rendering
- [ ] GET /api/analysis/tipping/{pitcher_id} — tipping analysis results
- [ ] GET /api/analysis/fatigue/{pitcher_id}/{outing_id} — fatigue tracking for an outing
- [ ] GET /api/analysis/command/{pitcher_id} — command analysis
- [ ] GET /api/analysis/arm-slot/{pitcher_id} — arm slot drift history
- [ ] GET /api/analysis/timing/{pitch_id} — kinetic chain timing for a pitch
- [ ] GET /api/analysis/baseline/{pitcher_id}/{pitch_type} — baseline comparison
- [ ] GET /api/compare — compare two pitches side-by-side (joint data + features)
- [ ] Demo mode: /api/demo/* mirrors all endpoints but reads from pre-computed data
- [ ] CORS configured for Next.js frontend
- [ ] OpenAPI/Swagger docs auto-generated

## Technical Approach

FastAPI with Pydantic response models. Background tasks for video processing (FastAPI BackgroundTasks or Celery for heavier workloads). WebSocket for progress streaming. Response serialization: JSON for metadata/analysis, binary (MessagePack or raw numpy) for large joint arrays to keep transfer fast.

Demo mode: a middleware or route prefix that swaps the data source from live database to static JSON/Parquet files.

## Files

| File | Purpose |
|------|---------|
| backend/app/main.py | FastAPI app, CORS, middleware |
| backend/app/api/routes.py | Route definitions |
| backend/app/api/pitchers.py | Pitcher endpoints |
| backend/app/api/pitches.py | Pitch data endpoints |
| backend/app/api/analysis.py | Analysis result endpoints |
| backend/app/api/upload.py | Video upload + job management |
| backend/app/api/compare.py | Pitch comparison endpoint |
| backend/app/api/demo.py | Demo mode data serving |
| backend/app/api/websocket.py | WebSocket progress streaming |
| backend/tests/test_api.py | API endpoint tests |

## Tasks

1. Set up FastAPI project with CORS and OpenAPI docs
2. Implement video upload + background processing job system
3. Build pitcher and pitch data endpoints
4. Build analysis result endpoints (all 6 modules)
5. Build pitch comparison endpoint
6. Implement WebSocket progress streaming
7. Build demo mode middleware (static data fallback)

## Dependencies

- Upstream: data-model, all 6 analysis modules
- Downstream: 3d-visualization, dashboard-ui
