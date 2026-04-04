## Test Review
P0: 0 | P1: 0 | P2: 1 | P3: 2

### P2: WebSocket endpoint untested
WebSocket /api/ws/jobs/{job_id} has no test. Requires async client (e.g., starlette.testclient WebSocket context). Not blocking since behavior is simple (poll + forward), but it's a real endpoint.

### P3: Upload file path not cleaned up in tests
test_upload_video creates a real file at ./data/videos/{pitcher_id}/. Tests leave behind a directory. Should use tmp_path monkeypatching or mock the file write.

### P3: No test for demo 503 when demo dir missing
GET /api/demo/pitchers should return 503 when ./data/demo/ doesn't exist. This branch is untested.

### Positives
- Fixtures use tmp_path — no shared state between tests
- Storage injection via app.state override is clean
- LLM mock tests are thorough (called, fallback, init all tested)
- Template functions each get their own test
- Edge cases covered: 404 for missing pitcher, pitch, job; compare with nonexistent pitch
