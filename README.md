# 🎥 Video Analytics — Detection · Tracking · Analytics

> Turn raw video into structured insight: detect objects with **YOLO**, track them across frames with persistent IDs, and compute **line-crossing counts, ROI dwell times, movement heatmaps, trajectories, and speed** — viewable in a **Streamlit dashboard**.

<p>
  <a href="https://github.com/stellasdeutsch-dev/video-analytics/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/stellasdeutsch-dev/video-analytics/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%E2%80%933.12-blue">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
  <img alt="Status" src="https://img.shields.io/badge/status-portfolio%20project-orange">
</p>

A real streaming-style data pipeline: video → detect → track → analytics → structured time-series → dashboard. Every detection becomes a data point. Heavy inference (YOLO) runs on a GPU cluster via **SLURM**; the tracking + analytics core is **pure NumPy** and runs (and is tested) anywhere — no GPU required.

📄 Full design & phased build plan: **[PROJECT_PLAN.md](PROJECT_PLAN.md)**

---

## ✨ Features

- **Detection** — Ultralytics **YOLO** (v11), any class set (vehicles, people, …).
- **Tracking** — IoU/SORT-style tracker with persistent IDs and a track lifecycle (pure NumPy; ByteTrack/BoT-SORT/DeepSORT are drop-in upgrades).
- **Analytics** — line-crossing counts **with direction**, ROI **zone dwell & occupancy**, movement **heatmap**, **trajectories**, and **speed** (px/s, or m/s with a scale).
- **Time-series output** — detections/tracks/analytics persisted as parquet + JSON.
- **Dashboard** — Streamlit: counts, zone stats, heatmap, active-tracks-over-time.
- **Annotated video** — optional rendered MP4 with boxes, IDs, lines, and zones.
- **Evaluation** — counting accuracy vs ground truth (MOTA/IDF1/mAP hooks documented).

---

## 🏗️ Architecture

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

Stages are independent modules sharing a parquet contract, so you can run the whole
thing (`src.pipeline`) or any single stage, and swap the tracker/detector freely.

---

## 🧰 Tech stack

| Layer | Choice | Why |
|---|---|---|
| Detection | **Ultralytics YOLO11** | SOTA speed/accuracy, trivial to run + fine-tune. |
| Tracking | **IoU/SORT-style** (pure NumPy) | Dependency-free + testable; shows the internals. ByteTrack/BoT-SORT are drop-in. |
| Analytics | **NumPy / pandas** | Counts, zones, heatmaps, speed as structured data. |
| Dashboard | **Streamlit** | Fast, data-app friendly. |
| Video I/O | **OpenCV** | Decode + annotated-video rendering. |
| Tooling | `pip`, `ruff`, `pytest`, YAML configs | Reproducible, config-driven. |

---

## 🚀 Quickstart (local, no GPU, no video)

> Use Python **3.10–3.12**. The synthetic demo exercises track → analytics → eval → dashboard with zero heavy deps for the core.

```bash
git clone <your-repo-url> && cd video-analytics
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"          # or: pip install -r requirements.txt

make sample        # synthetic detections (3 vehicles: 2 cross a line, 1 parks in a zone)
make track         # assign persistent track IDs
make analytics     # counts / zones / heatmap / speed -> data/run/analytics/
make eval          # counting accuracy vs eval/ground_truth.example.json
make dashboard     # Streamlit at http://localhost:8501
```

Run the unit tests (pure NumPy — no GPU/torch/cv2):

```bash
make test
```

---

## 🖥️ Full pipeline on the GPU cluster (real video)

Detection is the heavy step — run it on the university GPU via SLURM (cluster specifics in [PROJECT_PLAN.md](PROJECT_PLAN.md)).

```bash
# point configs/pipeline.yaml:source at your video and set the lines/zones
sbatch slurm/detect.slurm                      # detect -> track -> analytics (+ annotated.mp4)
# bring data/run/ back to your Mac, then:
VIDANALYTICS_OUT=data/run/analytics streamlit run web/dashboard.py
```

Or run stages separately: `python -m src.detect ...` → `python -m src.track ...` → `python -m src.analytics ...`.
Datasets to try: **UA-DETRAC**, **VisDrone**, **MOT17/20**, or your own footage.

### Compare YOLO models (speed vs. accuracy)

Benchmark several detectors on the same clip — FPS, detection volume, confidence, track count, and line-crossing counts side by side:

```bash
# edit configs/compare.yaml:models (e.g. yolo11n/s/m/x), then on the GPU:
sbatch slurm/compare.slurm          # or: python -m src.benchmark --config configs/compare.yaml
```

Writes `data/compare/comparison.md` (a Markdown table), `comparison.parquet`, and a `comparison.png` FPS chart — paste the table into the results section to show the speed/accuracy trade-off.

---

## 📁 Repo structure

```
configs/      detect / track / analytics / pipeline YAML
src/          ingest · detect (YOLO) · track (SORT) · analytics · annotate · pipeline · common
web/          Streamlit dashboard
slurm/        hello + detect SLURM jobs (GPU)
eval/         counting-accuracy harness + example ground truth
tests/        unit tests (geometry, tracker, analytics, integration) — no GPU needed
scripts/      synthetic-detections generator
PROJECT_PLAN.md   full phased plan & design
```

---

## 📊 Results

Fill in after running `eval/evaluate.py` (counting) and your MOT/COCO sets (mAP / MOTA / IDF1):

| Video | Detector | Tracker | Count MAE | Count acc. | mAP | MOTA | IDF1 | FPS (GPU) |
|---|---|---|---|---|---|---|---|---|
| _e.g. UA-DETRAC_ | _YOLO11s_ | _SORT_ | _—_ | _—_ | _—_ | _—_ | _—_ | _—_ |

---

## 🗺️ Roadmap

- [ ] Swap in ByteTrack / BoT-SORT and compare MOTA/IDF1
- [ ] Detector fine-tuning on the target domain (report mAP lift)
- [ ] Multi-camera / multi-line analytics
- [ ] RTSP live-stream mode
- [ ] TensorRT export for higher FPS
- [ ] Speed calibration via homography

---

## 📜 License

MIT — see [LICENSE](LICENSE).
