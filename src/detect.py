"""Detection stage — run a YOLO model (Ultralytics) over a video and write a
detections table. This is the heavy GPU step: run on the cluster via
slurm/detect.slurm. Locally it works on CPU/MPS for a short clip (smoke only).

Stage CLI:  python -m src.detect --config configs/detect.yaml
"""

from __future__ import annotations

import argparse
import logging
import time

from .common import load_config, save_json, select_device, setup_logging

log = logging.getLogger("vidanalytics")


def run(cfg: dict):
    import pandas as pd
    from ultralytics import YOLO

    device = select_device(cfg.get("device", "auto"))
    model = YOLO(cfg["model"])
    names = model.names
    log.info("Detecting with %s on %s over %s", cfg["model"], device, cfg["source"])

    results = model.predict(
        source=cfg["source"],
        stream=True,
        conf=cfg.get("conf", 0.25),
        iou=cfg.get("iou", 0.7),
        imgsz=cfg.get("imgsz", 640),
        classes=cfg.get("classes"),
        device=device,
        half=cfg.get("half", False) and str(device).startswith("cuda"),
        verbose=False,
    )

    rows = []
    n_frames = 0
    width = height = 0
    t0 = time.time()
    for frame_idx, r in enumerate(results):
        n_frames = frame_idx + 1
        if r.orig_shape:
            height, width = int(r.orig_shape[0]), int(r.orig_shape[1])
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cid = int(box.cls)
            rows.append({"frame": frame_idx, "class_id": cid, "class_name": names.get(cid, str(cid)),
                         "score": float(box.conf), "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        if frame_idx % 100 == 0 and frame_idx:
            log.info("frame %d  (%.1f fps)", frame_idx, n_frames / (time.time() - t0))

    df = pd.DataFrame(rows, columns=["frame", "class_id", "class_name", "score",
                                     "x1", "y1", "x2", "y2"])
    df.to_parquet(cfg["detections_path"], index=False)
    fps_proc = n_frames / max(time.time() - t0, 1e-9)
    log.info("Detected %d boxes over %d frames (%.1f fps) -> %s",
             len(df), n_frames, fps_proc, cfg["detections_path"])

    if cfg.get("meta_path"):
        fps = cfg.get("video_fps", 30.0)
        try:
            from .ingest import probe
            fps = probe(cfg["source"]).get("fps", fps) or fps
        except Exception as e:  # non-file source / no OpenCV -> fall back to config
            log.warning("could not probe fps from %s (%s); using %s", cfg["source"], e, fps)
        save_json({"fps": fps, "width": width, "height": height,
                   "n_frames": n_frames}, cfg["meta_path"])
    return df


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="YOLO detection over a video.")
    ap.add_argument("--config", default="configs/detect.yaml")
    ap.add_argument("--source", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.source:
        cfg["source"] = args.source
    if args.device:
        cfg["device"] = args.device
    run(cfg)


if __name__ == "__main__":
    main()
