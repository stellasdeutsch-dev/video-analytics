"""Video ingest — probe a video for metadata (fps, size, frame count) and
optionally extract frames. Uses OpenCV.

CLI:  python -m src.ingest --config configs/detect.yaml --meta data/video_meta.json
"""

from __future__ import annotations

import argparse
import logging

from .common import load_config, save_json, setup_logging

log = logging.getLogger("vidanalytics")


def probe(video_path: str) -> dict:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {video_path}")
    meta = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS)) or 30.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return meta


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="Probe a video for metadata.")
    ap.add_argument("--config", default="configs/detect.yaml")
    ap.add_argument("--source", default=None)
    ap.add_argument("--meta", default="data/video_meta.json")
    args = ap.parse_args()
    cfg = load_config(args.config)
    source = args.source or cfg["source"]
    meta = probe(source)
    save_json(meta, args.meta)
    log.info("Probed %s -> %s", source, meta)


if __name__ == "__main__":
    main()
