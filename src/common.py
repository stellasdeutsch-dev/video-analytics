"""Shared utilities: config/IO, device selection, and pure-NumPy geometry
(IoU, centroids, point-in-polygon, segment intersection) used by the tracker
and analytics. Geometry is dependency-free so it is testable without torch/cv2."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

LOGGER_NAME = "vidanalytics"
log = logging.getLogger(LOGGER_NAME)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(LOGGER_NAME)


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as e:  # pragma: no cover
        raise ImportError("PyYAML is required to load configs (pip install pyyaml)") from e
    with open(path) as f:
        return yaml.safe_load(f)


def save_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def select_device(pref: str = "auto") -> str:
    pref = (pref or "auto").lower()
    if pref != "auto":
        return pref
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


# ---------- geometry (pure NumPy) ----------

def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes a (N,4) and b (M,4), xyxy. Returns (N,M)."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def centroids(boxes: np.ndarray) -> np.ndarray:
    """Box centers from xyxy boxes (N,4) -> (N,2)."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0], axis=1)


def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Ray-casting point-in-polygon, vectorized over points. Returns bool (P,)."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    poly = np.asarray(polygon, dtype=float).reshape(-1, 2)
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cross = (yi > y) != (yj > y)              # excludes horizontal edges (yi == yj)
        denom = (yj - yi) if (yj - yi) != 0 else 1.0
        x_int = (xj - xi) * (y - yi) / denom + xi
        inside ^= cross & (x < x_int)
        j = i
    return inside


def orient(a, b, c) -> float:
    """Signed area*2 of triangle abc; >0 left turn, <0 right turn."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(p1, p2, p3, p4) -> bool:
    """True if segment p1p2 properly crosses segment p3p4."""
    d1 = orient(p3, p4, p1)
    d2 = orient(p3, p4, p2)
    d3 = orient(p1, p2, p3)
    d4 = orient(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))
