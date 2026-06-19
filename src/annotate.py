"""Render an annotated video: draw boxes, track IDs, counting lines, and ROI
zones onto the source frames. Optional (needs OpenCV); used by the pipeline."""

from __future__ import annotations

import logging

log = logging.getLogger("vidanalytics")

_PALETTE = [(66, 135, 245), (245, 66, 90), (66, 245, 135), (245, 200, 66),
            (180, 66, 245), (66, 245, 233), (245, 132, 66)]


def annotate(source: str, tracks_df, out_path: str, lines=None, zones=None) -> str:
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    by_frame = {f: g for f, g in tracks_df.groupby("frame")}
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for line in lines or []:
            (ax, ay), (bx, by) = line["points"]
            cv2.line(frame, (int(ax), int(ay)), (int(bx), int(by)), (0, 0, 255), 2)
        for zone in zones or []:
            pts = np.array(zone["polygon"], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
        g = by_frame.get(idx)
        if g is not None:
            for r in g.itertuples():
                color = _PALETTE[int(r.track_id) % len(_PALETTE)]
                cv2.rectangle(frame, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), color, 2)
                cv2.putText(frame, f"{r.class_name} #{int(r.track_id)}", (int(r.x1), int(r.y1) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()
    log.info("Wrote annotated video -> %s (%d frames)", out_path, idx)
    return out_path
