CONFIG_DIR ?= configs

.PHONY: help setup sample detect track analytics pipeline dashboard eval test lint smoke clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-11s\033[0m %s\n", $$1, $$2}'

setup:  ## Install the package with all extras
	python -m pip install -e ".[all]"

sample:  ## Generate synthetic detections (no video/GPU needed)
	python scripts/make_sample_detections.py --out data/detections.parquet --meta data/video_meta.json

detect:  ## Detection over a video (GPU; needs ultralytics)
	python -m src.detect --config $(CONFIG_DIR)/detect.yaml

track:  ## Track detections into persistent IDs
	python -m src.track --config $(CONFIG_DIR)/track.yaml

analytics:  ## Compute counts / zones / heatmap from tracks
	python -m src.analytics --config $(CONFIG_DIR)/analytics.yaml

pipeline:  ## Full pipeline on a video (detect -> track -> analytics)
	python -m src.pipeline --config $(CONFIG_DIR)/pipeline.yaml

compare:  ## Benchmark multiple YOLO models (GPU; speed + counting)
	python -m src.benchmark --config $(CONFIG_DIR)/compare.yaml

dashboard:  ## Launch the Streamlit dashboard
	VIDANALYTICS_OUT=data/run/analytics streamlit run web/dashboard.py

eval:  ## Evaluate counting accuracy vs ground truth
	python -m eval.evaluate --summary data/run/analytics/summary.json --gt eval/ground_truth.example.json

test:  ## Run unit tests (pure NumPy — no GPU/torch/cv2)
	pytest -q

lint:  ## Lint with ruff
	ruff check .

smoke: sample track analytics eval  ## Local end-to-end on synthetic detections (no GPU)
	@echo "Smoke pipeline complete — try: make dashboard"

clean:  ## Remove generated artifacts
	rm -rf data/detections.parquet data/tracks.parquet data/video_meta.json \
		data/run eval/report.json
