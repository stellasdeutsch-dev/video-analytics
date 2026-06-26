"""End-to-end orchestration: video -> detect -> track -> analytics -> outputs
(+ optional annotated video). One command for the whole flow.

CLI:  python -m src.pipeline --config configs/pipeline.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .common import load_config, setup_logging
from . import analytics as analytics_mod
from . import detect as detect_mod
from . import track as track_mod

log = logging.getLogger("vidanalytics")


def run(cfg: dict) -> dict:
    import pandas as pd

    out_dir = Path(cfg.get("out_dir", "data/run"))
    out_dir.mkdir(parents=True, exist_ok=True)
    det_path = out_dir / "detections.parquet"
    trk_path = out_dir / "tracks.parquet"
    meta_path = out_dir / "video_meta.json"

    # 1) detect
    det_cfg = {**cfg.get("detect", {}), "source": cfg["source"],
               "detections_path": str(det_path), "meta_path": str(meta_path),
               "video_fps": cfg.get("video_fps", 30.0)}
    detect_mod.run(det_cfg)

    # 2) track  (detect already applied the class filter; track keeps only its own optional classes)
    track_mod.run({**cfg.get("track", {}), "detections_path": str(det_path),
                   "tracks_path": str(trk_path)})

    # 3) analytics
    tracks_df = pd.read_parquet(trk_path)
    fps = cfg.get("video_fps", 30.0)
    width = height = 0
    if meta_path.exists():
        from .common import load_json
        meta = load_json(meta_path)
        fps = meta.get("fps", fps) or fps
        width, height = meta.get("width", 0), meta.get("height", 0)
    summary = analytics_mod.run(tracks_df, cfg.get("analytics", {}), fps, width, height, str(out_dir / "analytics"))

    # 4) optional annotated video
    if cfg.get("annotate_video", False):
        from .annotate import annotate
        annotate(cfg["source"], tracks_df, str(out_dir / "annotated.mp4"),
                 lines=cfg.get("analytics", {}).get("lines"),
                 zones=cfg.get("analytics", {}).get("zones"))

    log.info("Pipeline complete -> %s", out_dir)
    return summary


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="Run the full video analytics pipeline.")
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--source", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.source:
        cfg["source"] = args.source
    run(cfg)


if __name__ == "__main__":
    main()
