"""Evaluate analytics against ground truth.

Counting accuracy is the headline metric here (the analytics product is the
counts). Ground-truth file (JSON):
    {"line_counts": {"main_road": 12}, "n_tracks": 8}

For full detection mAP and tracking MOTA/IDF1, plug in your labeled MOT/COCO
sets with `motmetrics` / Ultralytics val (noted in the README) — kept out of the
default path so this runs without those heavy deps.

Usage:  python -m eval.evaluate --summary data/run/analytics/summary.json --gt eval/ground_truth.example.json
"""

from __future__ import annotations

import argparse
import json
import logging

from src.common import setup_logging

log = logging.getLogger("vidanalytics")


def counting_metrics(pred_counts: dict, gt_counts: dict) -> dict:
    errors, abs_errors = [], []
    per_line = {}
    for name, gt in gt_counts.items():
        pred = pred_counts.get(name, 0)
        err = pred - gt
        errors.append(err)
        abs_errors.append(abs(err))
        per_line[name] = {"pred": pred, "gt": gt, "abs_error": abs(err),
                          "accuracy": round(1 - abs(err) / max(gt, 1), 4)}
    n = max(len(errors), 1)
    return {
        "per_line": per_line,
        "MAE": round(sum(abs_errors) / n, 4),
        "mean_signed_error": round(sum(errors) / n, 4),
        "overall_accuracy": round(sum(v["accuracy"] for v in per_line.values()) / n, 4),
    }


def evaluate(summary: dict, gt: dict) -> dict:
    pred_counts = {c["name"]: c["total"] for c in summary.get("line_counts", [])}
    report = {"counting": counting_metrics(pred_counts, gt.get("line_counts", {}))}
    if "n_tracks" in gt:
        pred_tracks = summary.get("n_tracks", 0)
        report["tracks"] = {"pred": pred_tracks, "gt": gt["n_tracks"],
                            "abs_error": abs(pred_tracks - gt["n_tracks"])}
    return report


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="Evaluate counting accuracy vs ground truth.")
    ap.add_argument("--summary", default="data/run/analytics/summary.json")
    ap.add_argument("--gt", default="eval/ground_truth.example.json")
    ap.add_argument("--out", default="eval/report.json")
    args = ap.parse_args()
    with open(args.summary) as f:
        summary = json.load(f)
    with open(args.gt) as f:
        gt = json.load(f)
    report = evaluate(summary, gt)
    print(json.dumps(report, indent=2))
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
