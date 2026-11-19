#!/bin/bash
# One-time Python environment for the math evaluation (vLLM 0.6.3, as pinned by the repo).
# Run on a login node from the math/ directory:  bash setup_env.sh
# If your cluster asks you to keep conda envs outside $HOME (inode/space quotas) or to use
# containers, follow its documentation instead and run the pip lines inside that env.
set -euo pipefail
ENV_NAME="${ENV_NAME:-rlvr}"

eval "$(conda shell.bash hook)"
conda create -y -n "$ENV_NAME" python=3.10
conda activate "$ENV_NAME"

pip install vllm==0.6.3 "transformers<4.48" "math-verify[antlr4_11_0]==0.6.0" \
    word2number Pebble sympy==1.12 antlr4-python3-runtime==4.11.1 timeout-decorator \
    jieba datasets matplotlib "huggingface_hub[cli]"
(cd examples/math_eval/latex2sympy && pip install -e . --use-pep517)

python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"
