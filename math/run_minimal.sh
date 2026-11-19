#!/bin/bash
# Minimal pass@k reproduction: Qwen2.5-7B (base) vs Qwen-2.5-7B-SimpleRL-Zoo (RL)
# on the first NUM_TEST_SAMPLE problems of one benchmark.
#
# Usage:
#   bash run_minimal.sh                     # full minimal run (20 problems, n=32)
#   bash run_minimal.sh --smoke             # smoke test (2 problems, n=2, 4096 tokens, base model only)
#
# Override any setting via env vars, e.g.:
#   MODELS_DIR=~/models MAX_TOKENS=4096 bash run_minimal.sh
#
# NOTE: math_eval.py runs `rm -rf <model_name_or_path>` after loading the model,
# so this script always copies each model to a scratch dir first.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$SCRIPT_DIR/examples/math_eval"

MODELS_DIR="${MODELS_DIR:-$HOME/models}"
# Format: <local dir name>=<HF repo id>
MODELS="${MODELS:-Qwen2.5-7B=Qwen/Qwen2.5-7B SimpleRL-Zoo-7B=hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo}"
BENCHMARK="${BENCHMARK:-math500}"
NUM_TEST_SAMPLE="${NUM_TEST_SAMPLE:-20}"
N_SAMPLING="${N_SAMPLING:-32}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
MAX_TOKENS="${MAX_TOKENS:-16000}"
SEED="${SEED:-1}"
TEMPLATE="${TEMPLATE:-qwen-boxed}"
OUT_ROOT="${OUT_ROOT:-$SCRIPT_DIR/minimal_out}"
CKPT_TMP="${CKPT_TMP:-/tmp/rlvr_ckpt_$USER}"

if [ "${1:-}" = "--smoke" ]; then
    echo ">>> Smoke test mode"
    MODELS="${MODELS%% *}"   # first model only
    NUM_TEST_SAMPLE=2
    N_SAMPLING=2
    MAX_TOKENS=4096
    OUT_ROOT="$OUT_ROOT/smoke"
fi

mkdir -p "$MODELS_DIR" "$OUT_ROOT" "$CKPT_TMP"
cd "$EVAL_DIR"

for entry in $MODELS; do
    name="${entry%%=*}"
    repo="${entry#*=}"
    src="$MODELS_DIR/$name"

    if [ ! -f "$src/config.json" ]; then
        echo ">>> Downloading $repo -> $src"
        huggingface-cli download "$repo" --local-dir "$src"
    fi

    ckpt="$CKPT_TMP/$name"
    rm -rf "$ckpt"
    echo ">>> Copying $src -> $ckpt (math_eval.py deletes the model dir it loads)"
    cp -r "$src" "$ckpt"

    start=$(date +%s)
    TOKENIZERS_PARALLELISM=false python -u math_eval.py \
        --model_name_or_path "$ckpt" \
        --data_name "$BENCHMARK" \
        --output_dir "$OUT_ROOT/$name" \
        --split test \
        --prompt_type "$TEMPLATE" \
        --num_test_sample "$NUM_TEST_SAMPLE" \
        --n_sampling "$N_SAMPLING" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --max_tokens_per_call "$MAX_TOKENS" \
        --seed "$SEED" \
        --start 0 --end -1 \
        --use_vllm \
        --save_outputs \
        --overwrite
    echo ">>> $name finished in $(( $(date +%s) - start ))s"
    rm -rf "$ckpt"
done

result_files=()
for entry in $MODELS; do
    name="${entry%%=*}"
    result_files+=("$name=$OUT_ROOT/$name/$BENCHMARK/test_${TEMPLATE}_${NUM_TEST_SAMPLE}_seed${SEED}_t${TEMPERATURE}_s0_e-1.jsonl")
done

python "$SCRIPT_DIR/passk_minimal.py" --plot "$OUT_ROOT/passk_${BENCHMARK}.png" "${result_files[@]}"
