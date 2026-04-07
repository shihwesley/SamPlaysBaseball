"""End-to-end test of the delivery_window detector on a Darvish clip.

Run:
    python scripts/test_delivery_window.py <video_path>

Default test target: data/clips/526517/inn1_ab4_p1_CH_b69a4fbd.mp4
(the in-play CH that has a broadcast cut after Correa hits it)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.pipeline.delivery_window import find_delivery_window, write_trimmed_clip


def main() -> None:
    if len(sys.argv) < 2:
        clip = Path("data/clips/526517/inn1_ab4_p1_CH_b69a4fbd.mp4")
    else:
        clip = Path(sys.argv[1])

    if not clip.exists():
        print(f"FAIL: {clip} not found")
        sys.exit(1)

    print(f"=== delivery_window test on {clip.name} ===")
    print()
    t0 = time.time()
    window = find_delivery_window(clip)
    elapsed = time.time() - t0

    print(f"detection time: {elapsed:.1f}s")
    print(f"result:")
    for k, v in window.to_dict().items():
        print(f"  {k}: {v}")

    print()
    print("per-sample debug:")
    for s in window.sampled:
        marker = "P" if s.pitcher_detected else "."
        bbox_str = (
            f"bbox=({s.bbox[0]},{s.bbox[1]},{s.bbox[2]},{s.bbox[3]}) score={s.score:.2f}"
            if s.pitcher_detected else ""
        )
        print(f"  frame {s.frame_index:4d} [{marker}] {bbox_str} {s.note}")

    if window.set_frame is None:
        print()
        print("NO SET FRAME DETECTED — skipping trim write")
        return

    # Write a trimmed clip to disk
    trimmed_path = clip.parent / f"{clip.stem}_trimmed.mp4"
    print()
    print(f"writing trimmed clip → {trimmed_path}")
    write_trimmed_clip(clip, trimmed_path, window.start_frame, window.end_frame)
    print(f"  source: {window.total_frames} frames @ {window.fps:.2f} fps")
    print(f"  trim:   [{window.start_frame}, {window.end_frame}) = {window.n_frames} frames")
    print(f"  duration: {window.duration_seconds:.2f}s")


if __name__ == "__main__":
    main()
