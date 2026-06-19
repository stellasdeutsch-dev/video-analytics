#!/bin/bash
#SBATCH --job-name=compare
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

# Benchmark several YOLO models on the same video (speed + counting accuracy).
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vidanalytics

srun python -m src.benchmark --config configs/compare.yaml
