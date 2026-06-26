"""Analytics over tracks: line-crossing counts (with direction), ROI zone dwell
& occupancy, movement heatmap, trajectories, and speed. Every track/detection
becomes a structured time-series record. Core functions are pure NumPy and
testable; run() wires them to parquet/JSON outputs.

Stage CLI:  python -m src.analytics --config configs/analytics.yaml
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from .common import load_config, load_json, orient, points_in_polygon, save_json, segments_intersect, setup_logging
from .schema import validate_tracks

log = logging.getLogger("vidanalytics")


# ---------- pure core ----------

def build_trajectories(frames, track_ids, cx, cy) -> dict[int, np.ndarray]:
    """Group centroids by track_id into ordered (T,3) arrays [frame, cx, cy]."""
    frames = np.asarray(frames)
    track_ids = np.asarray(track_ids)
    cx = np.asarray(cx, dtype=float)
    cy = np.asarray(cy, dtype=float)
    traj: dict[int, np.ndarray] = {}
    for tid in np.unique(track_ids):
        m = track_ids == tid
        pts = np.stack([frames[m].astype(float), cx[m], cy[m]], axis=1)
        traj[int(tid)] = pts[np.argsort(pts[:, 0])]
    return traj


def count_crossings(traj: dict[int, np.ndarray], line_a, line_b) -> tuple[int, int]:
    """Count proper line crossings across all trajectories. Returns (forward,
    backward) where 'forward' = segment ends on the positive side of a->b.
    Points landing exactly on the line are not counted, so direction handling
    stays symmetric."""
    fwd = bwd = 0
    for pts in traj.values():
        c = pts[:, 1:3]
        for k in range(len(c) - 1):
            if segments_intersect(c[k], c[k + 1], line_a, line_b):
                side = orient(line_a, line_b, c[k + 1])
                if side > 0:
                    fwd += 1
                elif side < 0:
                    bwd += 1
    return fwd, bwd


def zone_dwell(traj: dict[int, np.ndarray], polygon, fps: float) -> dict[int, dict]:
    """Per-track frames-in-zone and dwell time (seconds)."""
    out: dict[int, dict] = {}
    for tid, pts in traj.items():
        inside = points_in_polygon(pts[:, 1:3], polygon)
        frames_in = int(inside.sum())
        if frames_in:
            out[tid] = {"frames": frames_in, "dwell_s": round(frames_in / max(fps, 1e-9), 3)}
    return out


def accumulate_heatmap(cx, cy, grid, width: float, height: float) -> np.ndarray:
    """2D histogram of centroids into a grid (cols, rows). Returns (rows, cols).
    Centroids are clamped into the frame so edge/overflow points land in the
    boundary bins instead of being dropped; extents fall back to the data."""
    cols, rows = int(grid[0]), int(grid[1])
    cx = np.asarray(cx, dtype=float)
    cy = np.asarray(cy, dtype=float)
    w = float(width) if width else (float(cx.max()) if cx.size else 1.0)
    h = float(height) if height else (float(cy.max()) if cy.size else 1.0)
    w, h = max(w, 1.0), max(h, 1.0)
    cx = np.clip(cx, 0, w)
    cy = np.clip(cy, 0, h)
    hist, _, _ = np.histogram2d(cy, cx, bins=[rows, cols], range=[[0, h], [0, w]])
    return hist


def track_speeds(traj: dict[int, np.ndarray], fps: float, meters_per_pixel=None) -> dict[int, float]:
    """Mean speed per track (px/s, or m/s if meters_per_pixel given)."""
    scale = meters_per_pixel if meters_per_pixel else 1.0
    out: dict[int, float] = {}
    for tid, pts in traj.items():
        if len(pts) < 2:
            continue
        steps = np.diff(pts[:, 1:3], axis=0)
        dist_px = np.linalg.norm(steps, axis=1).sum()
        dframes = pts[-1, 0] - pts[0, 0]
        if dframes > 0:
            out[tid] = round(float(dist_px * scale * fps / dframes), 3)
    return out


# ---------- wiring ----------

def run(tracks_df, params: dict, fps: float, width: float, height: float, out_dir: str) -> dict:
    from pathlib import Path

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df = tracks_df.copy()
    df["cx"] = (df["x1"] + df["x2"]) / 2.0
    df["cy"] = (df["y1"] + df["y2"]) / 2.0
    traj = build_trajectories(df["frame"], df["track_id"], df["cx"], df["cy"])

    line_counts = []
    for line in params.get("lines", []):
        a, b = line["points"]
        fwd, bwd = count_crossings(traj, a, b)
        line_counts.append({"name": line["name"], "forward": fwd, "backward": bwd, "total": fwd + bwd})

    zone_stats = []
    for zone in params.get("zones", []):
        dwell = zone_dwell(traj, zone["polygon"], fps)
        dwell_vals = [d["dwell_s"] for d in dwell.values()]
        zone_stats.append({
            "name": zone["name"],
            "unique_tracks": len(dwell),
            "total_dwell_s": round(float(sum(dwell_vals)), 3),
            "avg_dwell_s": round(float(np.mean(dwell_vals)), 3) if dwell_vals else 0.0,
        })

    heat = accumulate_heatmap(df["cx"], df["cy"], params.get("heatmap_grid", [64, 36]), width, height)
    np.save(Path(out_dir) / "heatmap.npy", heat)

    speeds = track_speeds(traj, fps, params.get("meters_per_pixel"))

    # per-frame counts time-series (objects present per frame)
    ts = df.groupby("frame")["track_id"].nunique().rename("active_tracks").reset_index()
    ts.to_parquet(Path(out_dir) / "counts_timeseries.parquet", index=False)

    summary = {
        "fps": fps,
        "frame_size": [int(width), int(height)],
        "n_frames": int(df["frame"].max()) + 1 if len(df) else 0,
        "n_tracks": int(df["track_id"].nunique()) if len(df) else 0,
        "n_detections": int(len(df)),
        "line_counts": line_counts,
        "zone_stats": zone_stats,
        "avg_speed": round(float(np.mean(list(speeds.values()))), 3) if speeds else 0.0,
        "speed_unit": "m/s" if params.get("meters_per_pixel") else "px/s",
    }
    save_json(summary, Path(out_dir) / "summary.json")
    save_json(speeds, Path(out_dir) / "speeds.json")
    log.info("Analytics -> %s  (%d tracks, lines=%s)", out_dir, summary["n_tracks"],
             [c["total"] for c in line_counts])
    return summary


def run_from_config(cfg: dict) -> dict:
    import pandas as pd

    df = validate_tracks(pd.read_parquet(cfg["tracks_path"]))
    fps = cfg.get("fps", 30.0)
    width = height = 0
    if cfg.get("meta_path"):
        try:
            meta = load_json(cfg["meta_path"])
            fps = meta.get("fps", fps)
            width, height = meta.get("width", 0), meta.get("height", 0)
        except FileNotFoundError:
            log.warning("meta file %s not found; using config fps=%s", cfg["meta_path"], fps)
    if not width and len(df):
        width = float(df[["x1", "x2"]].to_numpy().max())
        height = float(df[["y1", "y2"]].to_numpy().max())
    return run(df, cfg, fps, width, height, cfg.get("out_dir", "data/analytics"))


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="Compute analytics from tracks.")
    ap.add_argument("--config", default="configs/analytics.yaml")
    args = ap.parse_args()
    run_from_config(load_config(args.config))


if __name__ == "__main__":
    main()
