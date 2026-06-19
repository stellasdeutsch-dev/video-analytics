"""Benchmark several YOLO models on the same video and compare them on speed
(FPS), detection volume, confidence, track count, and line-crossing counts.

Heavy (runs detection per model) — run on the GPU via slurm/compare.slurm.
The summary/markdown helpers are pure and unit-tested.

CLI:  python -m src.benchmark --config configs/compare.yaml
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from . import detect as detect_mod
from .analytics import build_trajectories, count_crossings
from .common import load_config, setup_logging
from .track import SimpleTracker

log = logging.getLogger("vidanalytics")


def summarize_model(model, n_detections, mean_score, n_frames, wall_s, n_tracks, line_totals) -> dict:
    """Pure: assemble one model's comparison row (computes FPS)."""
    return {
        "model": model,
        "n_detections": int(n_detections),
        "mean_score": round(float(mean_score), 4),
        "n_frames": int(n_frames),
        "wall_s": round(float(wall_s), 2),
        "fps": round(n_frames / wall_s, 2) if wall_s else 0.0,
        "n_tracks": int(n_tracks),
        "line_totals": dict(line_totals),
    }


def to_markdown(results: list[dict], lines: list[dict]) -> str:
    """Pure: render the comparison rows as a Markdown table."""
    line_names = [ln["name"] for ln in lines]
    head = ["model", "fps", "n_detections", "mean_score", "n_tracks"] + [f"count[{n}]" for n in line_names]
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for r in results:
        cells = [r["model"], r["fps"], r["n_detections"], r["mean_score"], r["n_tracks"]]
        cells += [r["line_totals"].get(n, 0) for n in line_names]
        out.append("| " + " | ".join(str(c) for c in cells) + " |")
    return "\n".join(out) + "\n"


def _track_and_count(df, track_cfg: dict, lines: list[dict]):
    tracker = SimpleTracker(track_cfg.get("iou_threshold", 0.3), track_cfg.get("max_age", 30),
                            track_cfg.get("min_hits", 3))
    rows = []
    for frame, g in df.sort_values("frame").groupby("frame", sort=True):
        dets = g[["x1", "y1", "x2", "y2", "score", "class_id"]].to_numpy(dtype=float)
        for tr in tracker.update(dets):
            rows.append((int(frame), int(tr[0]), (tr[1] + tr[3]) / 2, (tr[2] + tr[4]) / 2))
    if not rows:
        return 0, {ln["name"]: 0 for ln in lines}
    arr = np.asarray(rows, dtype=float)
    traj = build_trajectories(arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3])
    totals = {}
    for ln in lines:
        a, b = ln["points"]
        fwd, bwd = count_crossings(traj, a, b)
        totals[ln["name"]] = fwd + bwd
    return len(traj), totals


def _plot(results: list[dict], path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = [r["model"] for r in results]
    fps = [r["fps"] for r in results]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(models, fps, color="#4287f5")
    ax.set_ylabel("FPS")
    ax.set_title("YOLO model speed comparison")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=120)


def benchmark(cfg: dict) -> list[dict]:
    out_dir = Path(cfg.get("out_dir", "data/compare"))
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = cfg.get("lines", [])
    results = []
    for model in cfg["models"]:
        det_cfg = {
            "model": model, "source": cfg["source"],
            "detections_path": str(out_dir / f"det_{Path(model).stem}.parquet"),
            "conf": cfg.get("conf", 0.25), "iou": cfg.get("iou", 0.7),
            "imgsz": cfg.get("imgsz", 640), "classes": cfg.get("classes"),
            "device": cfg.get("device", "auto"), "half": cfg.get("half", False),
        }
        log.info("Benchmarking %s ...", model)
        t0 = time.time()
        df = detect_mod.run(det_cfg)
        wall = time.time() - t0
        n_frames = int(df["frame"].max()) + 1 if len(df) else 0
        mean_score = float(df["score"].mean()) if len(df) else 0.0
        n_tracks, totals = _track_and_count(df, cfg.get("track", {}), lines) if len(df) else (0, {})
        results.append(summarize_model(model, len(df), mean_score, n_frames, wall, n_tracks, totals))

    import pandas as pd
    pd.DataFrame(results).to_parquet(out_dir / "comparison.parquet", index=False)
    md = to_markdown(results, lines)
    (out_dir / "comparison.md").write_text(md)
    print(md)
    try:
        _plot(results, out_dir / "comparison.png")
    except Exception as e:  # pragma: no cover
        log.warning("plot skipped (%s)", e)
    log.info("Comparison -> %s", out_dir / "comparison.md")
    return results


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="Benchmark/compare multiple YOLO models.")
    ap.add_argument("--config", default="configs/compare.yaml")
    args = ap.parse_args()
    benchmark(load_config(args.config))


if __name__ == "__main__":
    main()
