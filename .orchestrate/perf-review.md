## Performance Review
P0: 0 | P1: 1 | P2: 1 | P3: 2

### P1: _load_by_module does N sequential queries (analysis.py, generator.py)
For a pitcher with 100 pitches, `_load_by_module` calls `load_analysis` 100 times — one SQLite query per pitch. StorageLayer has no batch query method. This is the primary perf concern for pitchers with large pitch histories. A single `WHERE pitcher_id = ? AND module = ?` query on `analysis_results` would collapse this to one query. Currently the module has no such method; adding it would require a StorageLayer change which is out of scope for this phase — deferred.

### P2: GET /api/pitches/{id} returns full joint arrays (potentially large)
A 60-frame pitch at T×127×3 float64 ≈ 180k floats ≈ 1.4MB JSON. The spec calls for this. No streaming or chunking. Acceptable for single pitch requests but note clients should not request many concurrently.

### P3: GET /api/compare loads both pitches fully into memory simultaneously
Two large joint arrays in memory at once. For T=90 pitches both sides: ~2.8MB combined. Fine for the use case (side-by-side visualization) but worth noting.

### P3: ReportGenerator._latest iterates pitches in reverse to find last result
```python
for pid in reversed(pitch_ids):
    results = self.storage.load_analysis(pid, module=module)
```
Same N-query issue as above. Acceptable for report generation (infrequent), not for hot paths.

### Positives
- Pitcher list uses SQL for distinct pitch types (not Python-level dedup)
- Pagination on /pitchers/{id}/pitches prevents unbounded result sets
- Demo storage is lazily initialized — no startup cost if demo not used
