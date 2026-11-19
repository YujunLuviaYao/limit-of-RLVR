#!/bin/bash
# Main experiment: base vs RL model, one array task (one GPU) per model.
# Submit from the math/ directory:
#   cd limit-of-RLVR/math && sbatch slurm/main_experiment.sbatch
# A task that hits the time limit can simply be resubmitted: finished chunks are skipped.
# To run only one model: sbatch --array=1 slurm/main_experiment.sbatch
#
# Cluster-specific lines below (account / partition / GPU type) are commented out:
# fill them in from your cluster's documentation.
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
##SBATCH --gres=gpu:a100:1          # if your cluster names GPU types; needs >= 24GB for 7B

set -euo pipefail

MODELS=(
    "Qwen2.5-7B=Qwen/Qwen2.5-7B"
    "SimpleRL-Zoo-7B=hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo"
)
ENTRY="${MODELS[$SLURM_ARRAY_TASK_ID]}"

# Python environment: either submit from a shell where it is already active
# (sbatch passes your environment to the job), or set CONDA_ENV, e.g.
#   CONDA_ENV=rlvr sbatch slurm/main_experiment.sbatch
if [ -n "${CONDA_ENV:-}" ]; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
fi

cd "${SLURM_SUBMIT_DIR:-.}"
export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME}/hf_cache}"

echo "Host: $(hostname)  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Python: $(which python)"
bash run_main.sh "${ENTRY%%=*}" "${ENTRY#*=}"
