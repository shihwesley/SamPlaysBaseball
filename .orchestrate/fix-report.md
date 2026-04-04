## Fix Report

### P1 Fixed: Path traversal in upload.py
Added `re.fullmatch(r"[A-Za-z0-9_\-]{1,64}", pitcher_id)` validation before using pitcher_id in file path. Returns HTTP 400 if invalid.

### P1 Fixed: N+1 query in analysis endpoints and generator
Added `StorageLayer.load_analysis_by_pitcher(pitcher_id, module)` method that performs a single SQL query on `analysis_results WHERE pitcher_id = ? AND module = ?`. Updated `_load_by_module` in analysis.py and `_latest` in generator.py to use it.

### Deferred (P2/P3)
- Demo endpoint code duplication — acceptable for current scope
- datetime.utcnow() deprecation — non-breaking on supported Python versions
- PDF temp file cleanup — acceptable for low-volume usage
- WebSocket test — deferred to follow-up sprint

Tests: 34 passed, 0 failed after fixes.
