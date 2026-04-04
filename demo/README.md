# SamPlaysBaseball Demo

This directory contains everything needed to run a self-contained demo of the
SamPlaysBaseball pitching biomechanics platform — no GPU required.

## What the demo shows

Three synthetic pitchers (Sam Torres, Jake Kim, Demo Pitcher) with pre-computed
joint motion data, biomechanical analysis, and 3D visualization. The data is
synthetic but structurally identical to real pipeline output.

For historical legends analysis, see `legends/README.md`.

---

## Quick start (Docker)

```bash
# 1. Generate synthetic demo data
python demo/generate_synthetic.py

# 2. Start the stack (API + frontend)
cd demo
docker compose up
```

- Frontend: http://localhost:3000
- API docs: http://localhost:8000/docs

Docker build takes 2–4 minutes on first run. Subsequent starts: under 30 seconds.

---

## Quick start (manual)

**Prerequisites:** Python 3.11+, Node 20+

```bash
# 1. Install Python deps
pip install -r requirements.txt

# 2. Generate synthetic data
python demo/generate_synthetic.py

# 3. Start API + frontend together
python demo/launcher.py
```

Or start them separately:

```bash
# Terminal 1 — API
uvicorn backend.app.main:app --reload

# Terminal 2 — Frontend
cd frontend
NEXT_PUBLIC_DEMO_MODE=true NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

---

## Using real processed data

If you have a GPU machine and real pitch video:

```bash
# On the GPU machine
python demo/process_samples.py \
  --input-dir /path/to/mp4s \
  --output-dir demo/data \
  --pitcher-id your_pitcher \
  --pitcher-name "First Last" \
  --handedness R \
  --pitch-type FF

# Copy output back to your demo machine
rsync -av gpu-machine:/path/to/demo/data/ ./demo/data/
```

---

## Demo pitchers

| Pitcher | Hand | Pitch Types | What it illustrates |
|---------|------|-------------|---------------------|
| Sam Torres | RHP | FF, SL, CH | Standard 3-pitch mix, fatigue over game |
| Jake Kim | LHP | FF, CU, CH | Left-handed mechanics, curveball consistency |
| Demo Pitcher 01 | RHP | FF, SL | Tipping detection example |

---

## Loading data into the API

The demo API (`/api/demo/*`) reads JSON files from the `demo/data/` directory directly.
No database migration required — just run `generate_synthetic.py` and start the server.

---

## Troubleshooting

**API returns 503 "Demo data directory not found"**
Run `python demo/generate_synthetic.py` first.

**Frontend shows "Failed to fetch"**
Check that `NEXT_PUBLIC_API_URL` points to the running API (default: `http://localhost:8000`).

**Docker build fails on npm run build**
Ensure `frontend/package.json` exists and `npm install` runs successfully outside Docker first.

**Port conflict**
Set `API_PORT=8001` or `FRONTEND_PORT=3001` before running `launcher.py`.
