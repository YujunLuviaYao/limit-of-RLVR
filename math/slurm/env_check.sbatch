#!/bin/bash
# Quick check that the environment works on a GPU node: runs the smoke test
# (2 problems, n=2) with the small Qwen2.5-0.5B model. Takes a few minutes.
# Submit from the math/ directory with the env active:
#   cd limit-of-RLVR/math && sbatch slurm/env_check.sbatch
# If compute nodes have no internet, first download the model on a login node:
#   huggingface-cli download Qwen/Qwen2.5-0.5B --local-dir $SCRATCH/models/Qwen2.5-0.5B
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
if [ -n "${CONDA_ENV:-}" ]; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
fi
cd "${SLURM_SUBMIT_DIR:-.}"
export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME}/hf_cache}"

nvidia-smi
python -c "import torch, vllm; print('torch', torch.__version__, 'cuda ok:', torch.cuda.is_available(), '| vllm', vllm.__version__)"

MODELS_DIR="${SCRATCH:-$HOME}/models" MODELS="Qwen2.5-0.5B=Qwen/Qwen2.5-0.5B" \
    OUT_ROOT="$PWD/envcheck_out" bash run_minimal.sh --smoke
