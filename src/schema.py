"""Explicit inter-stage data contracts for the detect -> track -> analytics
pipeline. Validators check column presence AND box/score values, so a schema or
value drift fails fast at the boundary instead of corrupting analytics three
stages downstream. These constants are the single source of truth for the
parquet schemas documented in PROJECT_PLAN.md."""

from __future__ import annotations

import numpy as np

DETECTION_COLUMNS = ["frame", "class_id", "class_name", "score", "x1", "y1", "x2", "y2"]
TRACK_COLUMNS = ["frame", "track_id", "x1", "y1", "x2", "y2", "score", "class_id", "class_name"]


def missing_columns(present, required) -> list:
    """Pure helper: which required columns are absent from `present`."""
    have = set(present)
    return [c for c in required if c not in have]


def validate_columns(df, required, name: str):
    miss = missing_columns(list(df.columns), required)
    if miss:
        raise ValueError(f"{name} is missing columns {miss}; has {list(df.columns)}")
    return df


def _validate_boxes(df, name: str):
    if len(df) == 0:
        return
    box = df[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
    if not np.isfinite(box).all():
        raise ValueError(f"{name} has non-finite box coordinates")
    if (df["x2"].to_numpy() < df["x1"].to_numpy()).any() or \
       (df["y2"].to_numpy() < df["y1"].to_numpy()).any():
        raise ValueError(f"{name} has degenerate boxes (x2<x1 or y2<y1)")
    s = df["score"].to_numpy(dtype=float)
    if ((s < 0) | (s > 1)).any():
        raise ValueError(f"{name} has scores outside [0, 1]")


def validate_detections(df):
    validate_columns(df, DETECTION_COLUMNS, "detections")
    _validate_boxes(df, "detections")
    return df


def validate_tracks(df):
    validate_columns(df, TRACK_COLUMNS, "tracks")
    _validate_boxes(df, "tracks")
    return df
