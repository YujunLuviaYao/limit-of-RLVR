#!/bin/bash
# Quick check that the Singularity environment works on a GPU node: runs the smoke test
# (2 problems, n=2) with the small Qwen2.5-0.5B model. Takes a few minutes.
# Submit from the math/ directory:
#   cd ~/limit-of-RLVR/math && sbatch slurm/env_check.sbatch
# Compute nodes may have no internet: download the model on a login node first:
#   huggingface-cli download Qwen/Qwen2.5-0.5B --local-dir $SCRATCH/models/Qwen2.5-0.5B
# Image / overlay / conda env paths: see slurm/singularity_env.sh (override via env vars).
#SBATCH --job-name=rlvr-envcheck
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/%x_%j.out
##SBATCH --account=YOUR_ACCOUNT
##SBATCH --partition=YOUR_GPU_PARTITION

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/singularity_env.sh
export OUT_ROOT="${OUT_ROOT:-$SCRATCH/rlvr/envcheck_out}"
echo "Host: $(hostname)  Image: $SIF  Overlay: $OVERLAY (ro)  Env: $CONDA_ENV"

run_in_container '
    nvidia-smi
    echo "Python: $(which python)"
    python -c "import torch, vllm; print(\"torch\", torch.__version__, \"cuda ok:\", torch.cuda.is_available(), \"| vllm\", vllm.__version__)"
    MODELS="Qwen2.5-0.5B=Qwen/Qwen2.5-0.5B" bash run_minimal.sh --smoke
'
