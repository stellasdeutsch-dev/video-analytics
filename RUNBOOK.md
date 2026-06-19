# Runbook — running Video Analytics on the ITEC GPU cluster

Step-by-step to take this from scaffold to **real metrics in the README**. Local
work (code, tests, dashboard) is on the Mac; detection runs on the GPU.

## 0. One-time setup (on the cluster)
```bash
ssh -p 2022 USER@gpu6.itec.aau.at           # VPN required; gpu6 = 2× RTX 6000 Ada
# install miniforge into NFS home (~) if you haven't already, then:
conda create -n vidanalytics python=3.11 -y && conda activate vidanalytics
git clone https://github.com/stellasdeutsch-dev/video-analytics ~/video-analytics
cd ~/video-analytics
pip install -r requirements.txt             # torch from the CUDA index if needed
mkdir -p logs                                # SLURM --output dir (tracked via .gitkeep, but ensure it exists)
sbatch slurm/hello.slurm && squeue -u $USER  # Phase 0: confirm GPU is visible
```

## 1. Get a dataset
- **UA-DETRAC** (traffic, cleanest first target), **VisDrone**, or **MOT17/20** — from `/shares/datasets` if present, else download into `~/video-analytics/data/videos/`.
- For speed, stage onto NVMe: `rsync -a /shares/datasets/<set>/ /fastlocal/$USER/videos/` and point `configs/pipeline.yaml:source` there. (NVMe is **not** backed up — copy results back to `~`.)
- Set your counting `lines` and `zones` in `configs/pipeline.yaml` for the camera view.

## 2. Run the pipeline (GPU)
```bash
sbatch slurm/detect.slurm                    # detect -> track -> analytics (+ annotated.mp4)
squeue -u $USER                              # watch; logs in logs/detect_<jobid>.out
```
Outputs land in `data/run/` (`detections.parquet`, `tracks.parquet`, `analytics/`, `annotated.mp4`).

## 3. Compare detectors (optional, GPU)
```bash
# edit configs/compare.yaml:models = [yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11x.pt]
sbatch slurm/compare.slurm                   # -> data/compare/comparison.md + comparison.png
```

## 4. Evaluate
```bash
# label a few line counts / track totals as ground truth, then:
python -m eval.evaluate --summary data/run/analytics/summary.json --gt eval/ground_truth.json
# detection mAP + tracking MOTA/IDF1: use Ultralytics `yolo val` and `motmetrics` on a labeled MOT/COCO subset
```

## 5. Bring results back & view locally
```bash
# from the Mac:
rsync -avP -e "ssh -p 2022" USER@gpu6.itec.aau.at:~/video-analytics/data/run ./data/
VIDANALYTICS_OUT=data/run/analytics streamlit run web/dashboard.py
```

## 6. Fill the README
- Paste `data/compare/comparison.md` into the **Results** table (FPS vs accuracy).
- Add the eval numbers (counting MAE/accuracy, mAP, MOTA/IDF1) and the inference FPS.
- Screenshot the dashboard + grab a few seconds of `annotated.mp4` as a GIF for the README.
- Write the short **error-analysis** note: where the detector/tracker fail (small/occluded objects, ID switches) and what you changed.
