# CLAUDE.md — build guide for this repo

This repo implements **Video Analytics** (detection → tracking → analytics). The
full phased plan is in [PROJECT_PLAN.md](PROJECT_PLAN.md) — read it before large changes.

## Conventions
- Python ≥3.10. Code in `src/`, configs in `configs/` (YAML), tests in `tests/`.
- Each stage is a module with a CLI: `python -m src.<stage> --config configs/<stage>.yaml`.
- Stage data contract (parquet): detections `[frame,class_id,class_name,score,x1,y1,x2,y2]`
  → tracks add `track_id`. Keep this stable across stages.
- `src/track.py` (tracker) and `src/analytics.py` (counts/zones/heatmap/speed) are
  **pure NumPy** and must stay runnable without torch/ultralytics/cv2 so tests pass anywhere.
- Config-driven: no hardcoded paths, models, lines, or zones.

## Where things run
- **Detection (YOLO) is the heavy step — run on the uni GPU via SLURM** (`slurm/`).
  Local Mac = code + the synthetic-detections smoke test + unit tests.
- Loop: edit on Mac → `pytest -q` → push → pull on GPU → `sbatch slurm/detect.slurm`
  → monitor (`squeue`, W&B) → copy results back → `make dashboard` locally.

## Don't commit
videos, frames, detections/tracks parquet, model weights, outputs, `.env` — see `.gitignore`.
