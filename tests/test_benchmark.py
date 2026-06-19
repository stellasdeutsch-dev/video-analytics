"""Benchmark summary/markdown helpers (pure — no GPU)."""

from src.benchmark import summarize_model, to_markdown


def test_summarize_computes_fps():
    s = summarize_model("yolo11n.pt", 100, 0.83, 120, 4.0, 5, {"main_road": 3})
    assert s["fps"] == 30.0
    assert s["n_detections"] == 100
    assert s["mean_score"] == 0.83
    assert s["line_totals"]["main_road"] == 3


def test_summarize_zero_wall_time():
    s = summarize_model("yolo11n.pt", 0, 0.0, 0, 0.0, 0, {})
    assert s["fps"] == 0.0


def test_markdown_table_has_header_and_rows():
    rows = [
        summarize_model("yolo11n.pt", 10, 0.5, 30, 1.0, 2, {"main_road": 1}),
        summarize_model("yolo11s.pt", 20, 0.6, 30, 2.0, 2, {"main_road": 1}),
    ]
    md = to_markdown(rows, [{"name": "main_road"}])
    assert "| model |" in md
    assert "count[main_road]" in md
    assert "yolo11n.pt" in md and "yolo11s.pt" in md
    assert md.count("\n") >= 4   # header + separator + 2 rows
