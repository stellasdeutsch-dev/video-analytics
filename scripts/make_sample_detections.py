"""Generate a synthetic detections table (no video / no GPU needed) so the
track -> analytics -> dashboard -> eval flow runs end-to-end locally.

Scenario (1280x720 @ 30fps): three "vehicles" — one crossing the line at y=360
downward, one upward, one parked inside the junction zone. Matches
eval/ground_truth.example.json (line total = 2, n_tracks = 3).

Usage:  python scripts/make_sample_detections.py --out data/detections.parquet --frames 120
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

W, H, FPS = 1280, 720, 30.0


def box(cx, cy, s=22):
    return cx - s, cy - s, cx + s, cy + s


def main() -> None:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/detections.parquet")
    ap.add_argument("--meta", default="data/video_meta.json")
    ap.add_argument("--frames", type=int, default=120)
    args = ap.parse_args()

    n = args.frames
    rows = []
    for f in range(n):
        t = f / (n - 1)
        objs = [
            (300, 100 + (620 - 100) * t),   # crosses line downward
            (900, 620 + (100 - 620) * t),   # crosses line upward
            (200, 200),                      # parked in the junction zone
        ]
        for cx, cy in objs:
            x1, y1, x2, y2 = box(cx, cy)
            rows.append({"frame": f, "class_id": 2, "class_name": "car", "score": 0.9,
                         "x1": x1, "y1": y1, "x2": x2, "y2": y2})

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    Path(args.meta).parent.mkdir(parents=True, exist_ok=True)
    with open(args.meta, "w") as fh:
        json.dump({"fps": FPS, "width": W, "height": H, "n_frames": n}, fh, indent=2)
    print(f"Wrote {len(df)} detections over {n} frames -> {args.out}  (+ {args.meta})")


if __name__ == "__main__":
    main()
