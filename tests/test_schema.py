"""Data-contract validators — column presence AND box/score value checks."""

import pandas as pd
import pytest

from src.schema import (
    DETECTION_COLUMNS,
    TRACK_COLUMNS,
    missing_columns,
    validate_detections,
    validate_tracks,
)


def _dets(n=2):
    return pd.DataFrame({"frame": range(n), "class_id": [2] * n, "class_name": ["car"] * n,
                         "score": [0.9] * n, "x1": [10] * n, "y1": [10] * n,
                         "x2": [50] * n, "y2": [50] * n})


def _tracks(n=2):
    d = _dets(n)
    d.insert(1, "track_id", range(n))
    return d[TRACK_COLUMNS]


def test_missing_columns_helper():
    assert missing_columns(["frame", "x1"], DETECTION_COLUMNS) == \
        ["class_id", "class_name", "score", "y1", "x2", "y2"]


def test_detections_ok_and_missing_column():
    validate_detections(_dets())
    with pytest.raises(ValueError):
        validate_detections(_dets().drop(columns=["score"]))


def test_detections_degenerate_box_rejected():
    d = _dets()
    d.loc[0, "x2"] = 5                       # x2 < x1
    with pytest.raises(ValueError):
        validate_detections(d)


def test_detections_score_out_of_range_rejected():
    d = _dets()
    d.loc[1, "score"] = 1.7
    with pytest.raises(ValueError):
        validate_detections(d)


def test_tracks_ok_and_empty_ok():
    validate_tracks(_tracks())
    validate_tracks(_tracks(0))              # empty output still valid (schema preserved)


def test_tracks_missing_track_id():
    with pytest.raises(ValueError):
        validate_tracks(_dets())             # no track_id column
