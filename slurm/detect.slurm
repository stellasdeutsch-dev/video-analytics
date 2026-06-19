#!/bin/bash
#SBATCH --job-name=detect
#SBATCH --gres=gpu:1            # gpu:2 to use both GPUs on gpu6 / gpu4
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

# Detection over video on the cluster. Confirm partition / gres names: sinfo
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vidanalytics

# Optional: stage video(s) onto fast local NVMe (NOT backed up)
# rsync -a /shares/datasets/<your-videos>/ /fastlocal/$USER/videos/

# Run the whole pipeline (detect -> track -> analytics), or just `-m src.detect`.
srun python -m src.pipeline --config configs/pipeline.yaml
