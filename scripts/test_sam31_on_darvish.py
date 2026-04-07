"""One-shot test: load SAM 3.1 from mlx-vlm and detect Darvish on a 2017 frame.

Run:
    python scripts/test_sam31_on_darvish.py

Expected output:
    - Model weights download to ~/.cache/huggingface/ (~1-2 GB) on first run
    - Detection result on frame 90 of the Darvish CH pitch we already inferred
    - Annotated PNG saved to data/clips/526517/_sam31_test.png
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

print("loading SAM 3.1 from mlx-vlm...")
t0 = time.time()

from mlx_vlm.models.sam3.generate import Sam3Predictor
from mlx_vlm.models.sam3_1.processing_sam3_1 import Sam31Processor
from mlx_vlm.utils import load_model, get_model_path

MODEL_ID = "mlx-community/sam3.1-bf16"

print(f"  resolving {MODEL_ID}...")
model_path = get_model_path(MODEL_ID)
print(f"  model path: {model_path}")

print("  loading model into MLX memory...")
model = load_model(model_path)
print("  loading processor...")
processor = Sam31Processor.from_pretrained(str(model_path))
predictor = Sam3Predictor(model, processor, score_threshold=0.3)

t1 = time.time()
print(f"model ready in {t1 - t0:.1f}s")

# Run on the Darvish frame we already extracted
frame_path = Path("data/clips/526517/_frame90_HOME.jpg")
if not frame_path.exists():
    print(f"FAIL: {frame_path} not found")
    sys.exit(1)

image = Image.open(frame_path).convert("RGB")
print(f"\nimage: {image.size}")

# Try a few different prompts to see how SAM 3.1 handles them
prompts = [
    "a baseball pitcher",
    "a person in a baseball uniform",
    "a person",
]

for prompt in prompts:
    print(f"\nprompt: {prompt!r}")
    t2 = time.time()
    result = predictor.predict(image, text_prompt=prompt)
    t3 = time.time()
    n = len(result.scores) if hasattr(result, "scores") and result.scores is not None else 0
    print(f"  inference: {(t3-t2)*1000:.0f} ms")
    print(f"  detections: {n}")
    if n == 0:
        continue
    boxes = result.boxes
    scores = result.scores
    # Sort by area descending and print top 5
    areas = []
    for i in range(n):
        x1, y1, x2, y2 = boxes[i]
        areas.append((x2 - x1) * (y2 - y1))
    order = sorted(range(n), key=lambda i: areas[i], reverse=True)
    for rank, idx in enumerate(order[:5]):
        x1, y1, x2, y2 = boxes[idx]
        print(f"    [{rank}] score={scores[idx]:.3f} bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) area={areas[idx]:.0f}")

# Visualize the best result for the "a baseball pitcher" prompt
print("\nrendering annotated image...")
result = predictor.predict(image, text_prompt="a baseball pitcher")
frame = np.array(image)[..., ::-1].copy()  # RGB -> BGR for cv2
n = len(result.scores) if hasattr(result, "scores") and result.scores is not None else 0
for i in range(n):
    x1, y1, x2, y2 = (int(v) for v in result.boxes[i])
    score = float(result.scores[i])
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(
        frame,
        f"{score:.2f}",
        (x1, max(y1 - 5, 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
    )
out = Path("data/clips/526517/_sam31_test.png")
cv2.imwrite(str(out), frame)
print(f"WROTE: {out}")
