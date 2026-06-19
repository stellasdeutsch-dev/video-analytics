"""End-to-end core: synthetic detections -> tracker -> analytics, with no
pandas/torch/cv2. Mirrors scripts/make_sample_detections.py and the example
ground truth (3 tracks, 2 line crossings)."""

import numpy as np

from src.analytics import build_trajectories, count_crossings, zone_dwell
from src.track import SimpleTracker


def _box(cx, cy, s=22):
    return [cx - s, cy - s, cx + s, cy + s, 0.9, 2]


def test_pipeline_core_counts_and_tracks():
    n = 120
    tr = SimpleTracker(iou_threshold=0.3, max_age=30, min_hits=3)
    rows = []
    for f in range(n):
        t = f / (n - 1)
        dets = [
            _box(300, 100 + (620 - 100) * t),   # crosses line downward
            _box(900, 620 + (100 - 620) * t),   # crosses line upward
            _box(200, 200),                      # parked in zone
        ]
        for o in tr.update(dets):
            cx = (o[1] + o[3]) / 2
            cy = (o[2] + o[4]) / 2
            rows.append((f, int(o[0]), cx, cy))

    arr = np.asarray(rows, dtype=float)
    traj = build_trajectories(arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3])

    assert len(traj) == 3                              # three distinct tracks
    fwd, bwd = count_crossings(traj, [0, 360], [1280, 360])
    assert fwd + bwd == 2                              # two vehicles cross the line

    poly = [[100, 100], [300, 100], [300, 300], [100, 300]]
    dwell = zone_dwell(traj, poly, fps=30.0)
    assert max(v["frames"] for v in dwell.values()) >= n - 5   # parked car dwells throughout
