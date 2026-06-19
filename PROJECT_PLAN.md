# Video Analytics (Detection + Tracking + Analytics)

> Process video → detect objects, track them across frames with persistent IDs, and extract analytics (counts, trajectories, dwell time, zone crossings, speed) → dashboard + annotated video.

This document is **both** the build brief (hand each phase to Claude Code) **and** the portfolio writeup (architecture, skills, metrics) for the project.

---

## 1. What it does

- **Detect** objects per frame with YOLO (vehicles, people, or any COCO class).
- **Track** them across frames with stable IDs (IoU/SORT-style; ByteTrack/BoT-SORT drop-in).
- **Analyze**: line-crossing counts (with direction), ROI zone dwell & occupancy, movement heatmaps, trajectories, and speed.
- **Serve**: a Streamlit dashboard + an optional annotated MP4.
- Every detection/track is persisted as **structured time-series data** (parquet/JSON).

---

## 2. Why this is a strong AI / Data Engineer project (CV signal)

- **Streaming-style data pipeline** — video → detect → track → analyze → store; stages decoupled by a parquet contract.
- **GPU throughput optimization** — batch/stream YOLO inference, FPS tuning, fp16.
- **Time-series data generation + storage** — counts/occupancy/trajectories as queryable tables.
- **Classic CV stack** — detection + multi-object tracking + geometric analytics.
- **Reproducible & tested** — config-driven stages, a pure-NumPy core with unit tests, SLURM jobs.

---

## 3. Architecture

```
  video ──► detect (YOLO, GPU) ──► detections.parquet
                                        │
                                        ▼
                                 track (IoU/SORT) ──► tracks.parquet  (+ persistent track_id)
                                        │
                                        ▼
                                 analytics ──► counts · zones · heatmap · speed · timeseries
                                        │                     │
                                        ▼                     ▼
                              annotated.mp4            Streamlit dashboard
```

Stage data contract (parquet):
- **detections**: `frame, class_id, class_name, score, x1, y1, x2, y2`
- **tracks**: detections + `track_id`

---

## 4. Tech stack (chosen, with rationale)

| Layer | Choice | Why |
|---|---|---|
| Detection | **Ultralytics YOLO11** (`yolo11n` → `s/m/x`) | Fast, accurate, easy to run/fine-tune. *Alt:* RT-DETR. |
| Tracking | **IoU/SORT-style** (pure NumPy) | Dependency-free, testable, shows the internals. *Upgrade:* ByteTrack/BoT-SORT (Ultralytics built-in) or DeepSORT. |
| Analytics | **NumPy + pandas** | Geometric counts/zones/heatmaps as structured data. |
| Dashboard | **Streamlit** | Data-app friendly, fast to build. |
| Video I/O | **OpenCV** | Decode + annotated-video rendering. |
| Tracking/Exp | **Weights & Biases** | Log detection throughput / experiments. |
| Tooling | `pip`, `ruff`, `pytest`, YAML | Reproducible, config-driven. |

---

## 5. Repo structure

```
video-analytics/
├── CLAUDE.md / PROJECT_PLAN.md / README.md
├── Makefile / pyproject.toml / requirements.txt
├── configs/   detect.yaml · track.yaml · analytics.yaml · pipeline.yaml
├── src/
│   ├── ingest.py        # probe video metadata (fps, size, frames)
│   ├── detect.py        # YOLO detection -> detections.parquet (GPU entrypoint)
│   ├── track.py         # SimpleTracker (IoU/SORT) -> tracks.parquet
│   ├── analytics.py     # counts / zones / heatmap / speed (pure NumPy core)
│   ├── annotate.py      # render annotated video (OpenCV)
│   ├── pipeline.py      # detect -> track -> analytics orchestration
│   └── common.py        # config/IO, device, geometry helpers
├── web/dashboard.py     # Streamlit dashboard
├── slurm/   hello.slurm · detect.slurm
├── eval/    evaluate.py · ground_truth.example.json
├── tests/   geometry · tracker · analytics · integration
└── scripts/ make_sample_detections.py
```

---

## 6. Infrastructure & workflow (Mac M3 + ITEC GPU cluster)

### The development loop
1. **Write & test on the Mac M3** — code the tracker/analytics, run `pytest` (pure NumPy, no GPU).
2. **`git push`** → **`git pull`** on the GPU server.
3. **`sbatch slurm/detect.slurm`** to run YOLO over the real video(s).
4. **Monitor** via `squeue -u $USER` + W&B (FPS/throughput).
5. **Copy results back** → `make dashboard` locally.

### ITEC / AAU cluster specifics
- **Connect (VPN):** `ssh -p 2022 USER@gpu6.itec.aau.at`
- **Server choice:** `gpu6` (2× RTX 6000 Ada → fastest, default); `gpu4` (2× RTX 8000, 48 GB) for big models; `gpu3`/`gpu5` were locked until **2026-06-21** / **2026-06-29**.
- **Env:** miniforge in NFS `~` (backed up); code synced by `git pull`.
- **Data:** videos on `/shares/datasets` (10 TB NFS). Stage hot working set to `/fastlocal/$USER` (NVMe, **not backed up** → copy outputs back to `~`).
- **All GPU work via SLURM** (`sbatch`).

---

## 7. SLURM job template

`slurm/detect.slurm` — confirm partition/`gres` names with `sinfo`:

```bash
#!/bin/bash
#SBATCH --job-name=detect
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

source ~/miniforge3/etc/profile.d/conda.sh
conda activate vidanalytics
srun python -m src.pipeline --config configs/pipeline.yaml
```

---

## 8. Dataset options (pick by use case)

| Use case | Dataset | Notes |
|---|---|---|
| Traffic (cleanest) | **UA-DETRAC** | Vehicle detection + tracking, counts/speed. |
| Aerial / crowded | **VisDrone** | Small objects, dense scenes (harder). |
| Tracking benchmark | **MOT17 / MOT20** | Standard MOTA/IDF1 evaluation. |
| Your own | footage | Define your own lines/zones in the configs. |

Traffic is the cleanest first target (vehicle counts, trajectories, speed).

---

## 9. Phased execution plan

Each phase is a discrete task you can hand to Claude Code. "Where" = local vs GPU.

### Phase 0 — Scaffold · *local*
Repo, env, configs, `ruff`/`pytest`, W&B; submit `slurm/hello.slurm`.
**Done when:** `make test` passes locally and the hello GPU job prints the device.

### Phase 1 — Data & problem · *local → cluster*
Pick a use case (traffic). `src/ingest.py` probes video metadata; stage videos on the cluster.
**Done when:** video metadata is produced and a clip is ready to process.

### Phase 2 — Detection · *GPU*
`src/detect.py` runs YOLO over the video; evaluate on your domain; optionally fine-tune; log FPS.
**Done when:** detections written with measured mAP + FPS and annotated frames look right.

### Phase 3 — Tracking · *local*
`SimpleTracker` assigns persistent IDs; (optional) swap in ByteTrack/BoT-SORT and compare.
**Done when:** track IDs are stable and trajectories render on video.

### Phase 4 — Analytics · *local*
Line-crossing counts (direction), ROI dwell/occupancy, heatmap, speed → structured outputs.
**Done when:** analytics are computed and stored as parquet/JSON.

### Phase 5 — Pipeline orchestration · *local/GPU*
`src/pipeline.py`: one command video → detect → track → analytics → outputs (+ annotated MP4). Tune throughput; *stretch:* RTSP stream mode, TensorRT export.
**Done when:** a single command processes a video end to end.

### Phase 6 — Dashboard · *local*
Streamlit: counts, zone stats, heatmap, active-tracks time-series, annotated playback.
**Done when:** the dashboard shows analytics for a run.

### Phase 7 — Evaluation & error analysis · *local*
Counting accuracy vs ground truth; detection mAP; tracking MOTA/IDF1 (via `motmetrics`/Ultralytics val). Error analysis: missed small/occluded objects, ID switches — where and why.
**Done when:** an eval report + written error analysis exist.

### Phase 8 — Ship · *local*
README, architecture diagram, demo video/GIF, metrics table; optimization/deployment notes.
**Done when:** repo is portfolio-ready and runs from the README.

---

## 10. Evaluation & metrics to report

- **Detection:** mAP@50, mAP@50-95 (on a labeled subset).
- **Tracking:** MOTA, IDF1, ID switches, track fragmentation.
- **Counting:** MAE / accuracy of line counts vs ground truth.
- **Throughput:** inference FPS (detector size, fp16, batch).

---

## 11. Stretch goals

- ByteTrack/BoT-SORT/DeepSORT comparison; detector fine-tuning (mAP lift).
- Multi-camera tracking; anomaly detection; RTSP live mode; TensorRT speedup.
- Speed calibration via homography (real m/s).

---

## 12. Deliverables checklist (for the CV)

- [ ] Public GitHub repo, clean README, reproducible `make` targets.
- [ ] Demo (annotated video/GIF + dashboard screenshot).
- [ ] Metrics table (mAP / MOTA / IDF1 / counting accuracy / FPS).
- [ ] Architecture diagram + error-analysis writeup.
- [ ] Résumé bullet: *"Built a video analytics pipeline (YOLO + tracking + counting/zones/heatmaps) processing N FPS on a SLURM GPU cluster; counting accuracy X%, MOTA Y%."*

---

## 13. Quickstart commands

```bash
# Local setup (Mac M3)
make setup

# Phase 0 — verify cluster
ssh -p 2022 USER@gpu6.itec.aau.at
sbatch slurm/hello.slurm && squeue -u $USER

# Local end-to-end on synthetic detections (no GPU)
make smoke && make dashboard

# Real video on the cluster
sbatch slurm/detect.slurm            # detect -> track -> analytics
python -m eval.evaluate --summary data/run/analytics/summary.json --gt eval/ground_truth.example.json
```

---

*Build order: Phase 0 → 6 for a working demo, then 7 → 8 to make it portfolio-grade. Each phase is self-contained — hand it to Claude Code one at a time.*
