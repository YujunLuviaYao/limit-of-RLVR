#!/bin/bash
# Main experiment: base vs RL model, one array task (one GPU) per model, run inside the
# Singularity container. Submit from the math/ directory:
#   cd ~/limit-of-RLVR/math && sbatch slurm/main_experiment.sbatch
# A task that hits the time limit can simply be resubmitted: finished chunks are skipped.
# To run only one model: sbatch --array=1 slurm/main_experiment.sbatch
# Image / overlay / conda env paths: see slurm/singularity_env.sh (override via env vars).
# Results go to $SCRATCH/rlvr/main_out (override with OUT_ROOT).
#SBATCH --job-name=rlvr-main
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/%x_%A_%a.out
##SBATCH --account=YOUR_ACCOUNT
##SBATCH --partition=YOUR_GPU_PARTITION
##SBATCH --gres=gpu:a100:1          # if a GPU type is needed; 7B needs >= 24GB

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/singularity_env.sh
export OUT_ROOT="${OUT_ROOT:-$SCRATCH/rlvr/main_out}"

MODELS=(
    "Qwen2.5-7B=Qwen/Qwen2.5-7B"
    "SimpleRL-Zoo-7B=hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo"
)
ENTRY="${MODELS[$SLURM_ARRAY_TASK_ID]}"
export MODEL_NAME="${ENTRY%%=*}" MODEL_REPO="${ENTRY#*=}"
echo "Host: $(hostname)  Model: $MODEL_NAME  Image: $SIF  Overlay: $OVERLAY (ro)  Env: $CONDA_ENV"

run_in_container '
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "Python: $(which python)"
    bash run_main.sh "$MODEL_NAME" "$MODEL_REPO"
'
