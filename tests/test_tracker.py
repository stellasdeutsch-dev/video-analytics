"""SimpleTracker — association, ID stability, and lifecycle (pure NumPy)."""

from src.track import SimpleTracker


def _box(cx, cy, s=20):
    return [cx - s, cy - s, cx + s, cy + s, 0.9, 2]


def test_single_object_keeps_one_id():
    tr = SimpleTracker(iou_threshold=0.3, max_age=5, min_hits=2)
    ids = set()
    for f in range(10):
        for o in tr.update([_box(100 + f * 3, 100)]):
            ids.add(int(o[0]))
    assert ids == {1}


def test_two_objects_get_two_ids():
    tr = SimpleTracker(iou_threshold=0.3, max_age=5, min_hits=1)
    out = None
    for f in range(6):
        out = tr.update([_box(100 + f * 3, 100), _box(500 + f * 3, 400)])
    assert len({int(o[0]) for o in out}) == 2


def test_lost_track_expires_then_new_id():
    tr = SimpleTracker(iou_threshold=0.3, max_age=2, min_hits=1)
    for _ in range(4):
        tr.update([_box(100, 100)])
    for _ in range(5):              # object disappears beyond max_age
        tr.update([])
    out = tr.update([_box(100, 100)])  # reappears -> fresh id
    assert int(out[0][0]) > 1


def test_no_detections_returns_empty():
    tr = SimpleTracker()
    assert tr.update([]).shape == (0, 7)
