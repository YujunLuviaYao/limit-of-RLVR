#!/bin/bash
# One-time Python environment for the math evaluation (vLLM 0.6.3, as pinned by the repo).
# Run on a login node from the math/ directory:
#   bash setup_env.sh                                  # env at $SCRATCH/envs/rlvr
#   ENV_PREFIX=/some/other/path bash setup_env.sh
# The env and pip cache go to $SCRATCH (not $HOME) because clusters usually have small
# home quotas. Activate later with:  conda activate $SCRATCH/envs/rlvr
# No CUDA module is needed: the vLLM/PyTorch wheels ship their own CUDA libraries and only
# need the GPU node's NVIDIA driver.
set -euo pipefail

BASE="${SCRATCH:-$HOME}"
ENV_PREFIX="${ENV_PREFIX:-$BASE/envs/rlvr}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$BASE/.cache/pip}"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found on PATH. Either load your cluster's conda module, or, for a"
    echo "Miniconda you installed yourself (e.g. in a Singularity overlay), put it on PATH:"
    echo "    export PATH=/ext3/miniconda3/bin:\$PATH"
    echo "Sourcing .../etc/profile.d/conda.sh alone is not enough: it only defines a shell"
    echo "function in the current shell, which this script (a child process) does not see."
    exit 1
fi

eval "$(conda shell.bash hook)"
if [ ! -d "$ENV_PREFIX" ]; then
    conda create -y -p "$ENV_PREFIX" python=3.10
fi
conda activate "$ENV_PREFIX"

pip install vllm==0.6.3 "transformers<4.48" "math-verify[antlr4_11_0]==0.6.0" \
    word2number Pebble sympy==1.12 antlr4-python3-runtime==4.11.1 timeout-decorator \
    jieba datasets matplotlib "huggingface_hub[cli]"
(cd examples/math_eval/latex2sympy && pip install -e . --use-pep517)

python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"
echo "Done. Activate with: conda activate $ENV_PREFIX"
