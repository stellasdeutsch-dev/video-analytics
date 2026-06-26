"""Multi-object tracking — an IoU/SORT-style tracker (pure NumPy, dependency-free
and testable). For production you can swap in Ultralytics' built-in ByteTrack /
BoT-SORT or DeepSORT; this keeps the data contract identical.

Stage CLI:  python -m src.track --config configs/track.yaml
Consumes detections.parquet -> writes tracks.parquet with persistent track_id.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from .common import iou_matrix, load_config, setup_logging
from .schema import TRACK_COLUMNS, validate_detections, validate_tracks

log = logging.getLogger("vidanalytics")


class SimpleTracker:
    """Greedy IoU association with a simple track lifecycle (hits / max_age)."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30, min_hits: int = 3) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: list[dict] = []
        self._next_id = 1
        self._frame = 0

    def update(self, dets) -> np.ndarray:
        """dets: (N,6) [x1,y1,x2,y2,score,class_id]. Returns (M,7)
        [track_id,x1,y1,x2,y2,score,class_id] for tracks updated this frame.

        A track not matched on a given frame emits nothing until re-matched, so a
        sparse/flickering detector can leave single-frame gaps in the per-frame
        active-track counts. Trajectory analytics (crossings, dwell, speed) are
        unaffected because build_trajectories only sorts the points it has."""
        self._frame += 1
        dets = np.asarray(dets, dtype=float).reshape(-1, 6) if dets is not None else np.zeros((0, 6))
        n_det = len(dets)

        matches: list[tuple[int, int]] = []
        if self.tracks and n_det:
            tboxes = np.array([t["box"] for t in self.tracks])
            ious = iou_matrix(tboxes, dets[:, :4]).copy()
            while ious.size:
                ti, di = divmod(int(np.argmax(ious)), ious.shape[1])
                if ious[ti, di] < self.iou_threshold:
                    break
                matches.append((ti, di))
                ious[ti, :] = -1.0
                ious[:, di] = -1.0

        matched_t = {t for t, _ in matches}
        matched_d = {d for _, d in matches}

        for ti, di in matches:
            t = self.tracks[ti]
            t["box"] = dets[di, :4]
            t["score"] = float(dets[di, 4])
            t["class_id"] = int(dets[di, 5])
            t["hits"] += 1
            t["tsu"] = 0

        for ti, t in enumerate(self.tracks):
            if ti not in matched_t:
                t["tsu"] += 1

        for di in range(n_det):
            if di not in matched_d:
                self.tracks.append({
                    "id": self._next_id, "box": dets[di, :4], "score": float(dets[di, 4]),
                    "class_id": int(dets[di, 5]), "hits": 1, "tsu": 0,
                })
                self._next_id += 1

        self.tracks = [t for t in self.tracks if t["tsu"] <= self.max_age]

        out = []
        for t in self.tracks:
            if t["tsu"] == 0 and t["hits"] >= self.min_hits:
                out.append([t["id"], *t["box"], t["score"], t["class_id"]])
        return np.asarray(out, dtype=float).reshape(-1, 7)


def run(cfg: dict):
    import pandas as pd

    df = validate_detections(pd.read_parquet(cfg["detections_path"])).sort_values("frame")
    names = {int(r.class_id): r.class_name for r in df[["class_id", "class_name"]].drop_duplicates().itertuples()}
    keep = cfg.get("classes")
    tracker = SimpleTracker(cfg.get("iou_threshold", 0.3), cfg.get("max_age", 30), cfg.get("min_hits", 3))

    rows = []
    for frame, g in df.groupby("frame", sort=True):
        if keep:
            g = g[g["class_id"].isin(keep)]
        dets = g[["x1", "y1", "x2", "y2", "score", "class_id"]].to_numpy(dtype=float)
        for tr in tracker.update(dets):
            tid, x1, y1, x2, y2, score, cid = tr
            rows.append({"frame": int(frame), "track_id": int(tid), "x1": x1, "y1": y1,
                         "x2": x2, "y2": y2, "score": score, "class_id": int(cid),
                         "class_name": names.get(int(cid), str(int(cid)))})

    out = pd.DataFrame(rows, columns=TRACK_COLUMNS)   # explicit cols keep schema when empty
    validate_tracks(out)
    out.to_parquet(cfg["tracks_path"], index=False)
    n_tracks = out["track_id"].nunique() if len(out) else 0
    log.info("Tracked %d detections into %d tracks -> %s", len(out), n_tracks, cfg["tracks_path"])
    return out


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="Track detections into persistent IDs.")
    ap.add_argument("--config", default="configs/track.yaml")
    args = ap.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
