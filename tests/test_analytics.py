"""Analytics core — crossings, dwell, heatmap, speed (pure NumPy)."""

import numpy as np

from src.analytics import (
    accumulate_heatmap,
    build_trajectories,
    count_crossings,
    track_speeds,
    zone_dwell,
)


def test_crossing_directions():
    frames = np.array([0, 1, 2, 0, 1, 2])
    tids = np.array([1, 1, 1, 2, 2, 2])
    cx = np.array([100, 100, 100, 500, 500, 500.0])
    cy = np.array([100, 300, 500, 500, 300, 100.0])   # t1 down, t2 up
    traj = build_trajectories(frames, tids, cx, cy)
    fwd, bwd = count_crossings(traj, [0, 360], [1280, 360])
    assert (fwd, bwd) == (1, 1)


def test_no_crossing():
    traj = build_trajectories(np.array([0, 1, 2]), np.array([1, 1, 1]),
                              np.array([10, 10, 10.0]), np.array([10, 20, 30.0]))
    assert count_crossings(traj, [0, 360], [1280, 360]) == (0, 0)


def test_zone_dwell():
    frames = np.array([0, 1, 2, 3])
    tids = np.array([1, 1, 1, 1])
    cx = np.array([150, 150, 150, 150.0])
    cy = np.array([150, 150, 150, 900.0])             # 3 inside, 1 outside
    traj = build_trajectories(frames, tids, cx, cy)
    poly = [[100, 100], [300, 100], [300, 300], [100, 300]]
    d = zone_dwell(traj, poly, fps=10.0)
    assert d[1]["frames"] == 3
    assert abs(d[1]["dwell_s"] - 0.3) < 1e-6


def test_heatmap_shape_and_mass():
    h = accumulate_heatmap(np.array([10, 20, 30.0]), np.array([10, 20, 30.0]), [4, 3], 100, 100)
    assert h.shape == (3, 4)
    assert h.sum() == 3


def test_speed_pixels_per_second():
    traj = build_trajectories(np.array([0, 1, 2]), np.array([1, 1, 1]),
                              np.array([0, 10, 20.0]), np.array([0, 0, 0.0]))
    s = track_speeds(traj, fps=1.0)                   # 20px / 2 frames * 1fps = 10 px/s
    assert abs(s[1] - 10.0) < 1e-6
