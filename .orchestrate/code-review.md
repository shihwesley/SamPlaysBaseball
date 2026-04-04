## Code Review
P0: 0 | P1: 1 | P2: 2 | P3: 3

### P1: Path traversal risk in upload.py
`pitcher_id` from form input is used directly in a path:
```python
video_dir = _VIDEO_DIR / pitcher_id
```
A malicious pitcher_id like `../../etc` would write outside the intended directory. Should sanitize: strip path separators, validate alphanumeric + hyphens only.

### P2: _build_summaries has N+1 pattern for pitch_count
For each pitcher, calls `storage.list_pitch_ids()` (one SQL query per pitcher) then another query for distinct pitch_types. With 50 pitchers this is 100+ queries. Both can be one aggregation query.

### P2: demo.py duplicates logic from real endpoints verbatim
pitchers.py `_build_summaries` is reimplemented in demo.py rather than extracted to a shared utility. Creates drift risk when real endpoints change.

### P3: datetime.utcnow() deprecation (Python 3.14)
upload.py and api/models.py use `datetime.utcnow()` which is deprecated. Should use `datetime.now(UTC)`. Not blocking for Python ≤3.12 targets.

### P3: upload.py doesn't validate pitch_type values
Any string is accepted as pitch_type. Could be constrained to known types (FF, SL, CH, etc.) but spec doesn't require it, so minor.

### P3: pdf.py temp files not cleaned up
`_serve_pdf` in reports.py creates a NamedTemporaryFile with `delete=False` but never deletes it after the response is served. Fine for low-volume usage, accumulates on disk over time.

### Positives
- Clean separation of deps.py to avoid circular imports
- All routes use parameterized SQL (no injection risk)
- Consistent use of `from __future__ import annotations`
- LLM fallback pattern is robust — never raises to caller
- Demo mode correctly isolated to separate storage instance
