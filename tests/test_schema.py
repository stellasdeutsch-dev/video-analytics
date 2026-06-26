"""Data-contract validators — pure (duck-typed on .columns, no pandas needed)."""

import pytest

from src.schema import (
    DETECTION_COLUMNS,
    TRACK_COLUMNS,
    missing_columns,
    validate_detections,
    validate_tracks,
)


class _DF:
    def __init__(self, cols):
        self.columns = cols


def test_missing_columns():
    assert missing_columns(["frame", "x1"], DETECTION_COLUMNS) == \
        ["class_id", "class_name", "score", "y1", "x2", "y2"]
    assert missing_columns(DETECTION_COLUMNS, DETECTION_COLUMNS) == []


def test_validate_detections():
    validate_detections(_DF(DETECTION_COLUMNS))                 # ok
    with pytest.raises(ValueError):
        validate_detections(_DF(["frame", "x1"]))


def test_validate_tracks():
    validate_tracks(_DF(TRACK_COLUMNS))                         # ok
    with pytest.raises(ValueError):
        validate_tracks(_DF(DETECTION_COLUMNS))                 # missing track_id
