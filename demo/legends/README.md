# Historical Legends — Footage Guide

This document covers how to acquire, prepare, and process historical pitcher footage
for the SamPlaysBaseball legends analysis.

**Honest disclaimer upfront:** SAM 3D Body was trained on controlled multi-camera
footage at 60fps. On 30fps broadcast video with compression artifacts, joint position
error increases from ~15mm RMS to ~25–40mm. This affects wrist and finger kinematics
significantly. Trunk, shoulder, and release point metrics remain interpretable.

---

## Supported legends

| Pitcher | ID | Hand | Era | Best footage era |
|---|---|---|---|---|
| Mariano Rivera | `rivera` | R | 1995–2013 | 2000–2013 (ESPN/YES) |
| Pedro Martinez | `pedro_martinez` | R | 1992–2009 | 2002–2004 (NESN/Fox) |
| Clayton Kershaw | `kershaw` | L | 2008–present | 2012–present (HD) |
| Greg Maddux | `maddux` | R | 1986–2008 | 1992–2003 (TBS/Fox) |
| Randy Johnson | `randy_johnson` | L | 1988–2009 | 1999–2004 (AZ) |

---

## Legal note

Only process footage you have the rights to use. Options:

1. **MLB Film Room** (mlb.com/video) — MLB's official archive. Individual clips require
   an MLB account. Commercial use requires a licensing agreement with MLB Advanced Media.

2. **MLB official YouTube channels** (MLB, team channels) — publicly posted highlight
   clips. Check video description for license terms before commercial use.

3. **Licensed MLB footage** — contact MLB.com/licensing for bulk licensing for a
   presentation or product demo.

Do not download and process broadcast footage from unofficial sources for any
commercial or public presentation.

---

## Video specifications

| Spec | Recommended | Minimum |
|---|---|---|
| Frame rate | 60fps | 24fps |
| Resolution | 1080p | 480p |
| Format | MP4 (H.264) | Any |
| Angle | Side view or 45-degree | Avoid head-on |
| Clip length | Full delivery (2–4 seconds) | Wind-up to release |

Side-view footage produces the most accurate arm slot and release point measurements.
Head-on camera angles are not suitable for biomechanical analysis.

---

## Processing steps

```bash
# Install dependencies on GPU machine
pip install -r requirements.txt

# Process Rivera cutter footage
python demo/legends/process_legends.py \
  --input-dir /path/to/rivera_clips \
  --pitcher-id rivera \
  --quality-flag broadcast \
  --device cuda \
  --output-dir demo/data/legends

# Process Kershaw (HD footage)
python demo/legends/process_legends.py \
  --input-dir /path/to/kershaw_clips \
  --pitcher-id kershaw \
  --quality-flag hd \
  --device cuda
```

Quality flags:
- `hd` — modern HD footage, minimal smoothing
- `broadcast` — standard 30fps broadcast (default, recommended for 1995–2013 footage)
- `archival` — low-quality archive footage, maximum smoothing

---

## Preparing clips

1. Trim each clip to a single pitch delivery (2–4 seconds, wind-up to follow-through)
2. One MP4 file per pitch
3. Name files descriptively: `rivera_cutter_001.mp4`, `maddux_ff_009.mp4`
4. 10–20 clips per pitcher is enough for meaningful baseline statistics

---

## Accuracy expectations by era

| Footage era | Resolution | Accuracy | Reliable metrics |
|---|---|---|---|
| 2015–present | 1080p 60fps | High | All metrics |
| 2010–2014 | 720p–1080p | Good | Arm slot, release point, stride |
| 2000–2009 | 480p broadcast | Moderate | Arm slot, release point trends |
| 1990–1999 | 480p or lower | Low | Arm slot only |

For Maddux and Rivera's 1990s footage, focus talking points on arm slot and
release point rather than fine wrist kinematics.

---

## Biomechanical talking points

See `talking_points.json` for per-legend analysis notes, expected metric values,
and what each pitcher's mechanics illustrate in the context of the analysis modules.
