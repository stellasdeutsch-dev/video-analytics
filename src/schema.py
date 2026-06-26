"""Explicit inter-stage data contracts for the detect -> track -> analytics
pipeline. Lightweight, dependency-free column checks so a schema drift in one
stage fails fast at the boundary with a clear message instead of surfacing as a
downstream KeyError on the GPU. These constants are the single source of truth
for the parquet schemas documented in PROJECT_PLAN.md."""

from __future__ import annotations

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


def validate_detections(df):
    return validate_columns(df, DETECTION_COLUMNS, "detections")


def validate_tracks(df):
    return validate_columns(df, TRACK_COLUMNS, "tracks")
