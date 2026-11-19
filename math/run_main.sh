#!/bin/bash
# Main experiment for ONE model: n samples per problem on a MATH500 subset, run in
# chunks so that an interrupted job can be resubmitted and resumes where it stopped
# (finished chunks are skipped).
#
# Usage:
#   bash run_main.sh <name> <hf_repo>
#   bash run_main.sh Qwen2.5-7B Qwen/Qwen2.5-7B
#
# Settings (env vars, defaults follow the paper's MATH500 setup):
#   DATA_DIR    dir containing math500/test.jsonl (default: subsets/math500_strat200_seed0)
#   N_SAMPLING  128      TEMPERATURE 0.6     TOP_P 0.95     MAX_TOKENS 16000
#   SEED 1      TEMPLATE qwen-boxed          CHUNK_SIZE 50
#   MODELS_DIR  where models are downloaded (default: $SCRATCH/models or ~/models)
#   OUT_ROOT    where results go (default: $SCRATCH/rlvr/main_out, or main_out/ without $SCRATCH)

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: bash run_main.sh <name> <hf_repo>"; exit 1
fi
NAME="$1"
REPO="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$SCRIPT_DIR/examples/math_eval"

DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/subsets/math500_strat200_seed0}"
BENCHMARK=math500
N_SAMPLING="${N_SAMPLING:-128}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
MAX_TOKENS="${MAX_TOKENS:-16000}"
SEED="${SEED:-1}"
TEMPLATE="${TEMPLATE:-qwen-boxed}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
MODELS_DIR="${MODELS_DIR:-${SCRATCH:-$HOME}/models}"
OUT_ROOT="${OUT_ROOT:-${SCRATCH:+$SCRATCH/rlvr/main_out}}"
OUT_ROOT="${OUT_ROOT:-$SCRIPT_DIR/main_out}"

DATA_FILE="$DATA_DIR/$BENCHMARK/test.jsonl"
if [ ! -f "$DATA_FILE" ]; then
    echo "Missing $DATA_FILE. Create it first: python make_subset.py"; exit 1
fi
NUM=$(wc -l < "$DATA_FILE")

MODEL_PATH="$MODELS_DIR/$NAME"
if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo ">>> Downloading $REPO -> $MODEL_PATH"
    huggingface-cli download "$REPO" --local-dir "$MODEL_PATH"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export KEEP_MODEL_DIR=1   # load the model in place; math_eval.py would otherwise delete it
export TOKENIZERS_PARALLELISM=false

echo ">>> $NAME: $NUM problems, n=$N_SAMPLING, chunks of $CHUNK_SIZE"
cd "$EVAL_DIR"
for ((s = 0; s < NUM; s += CHUNK_SIZE)); do
    e=$(( s + CHUNK_SIZE < NUM ? s + CHUNK_SIZE : NUM ))
    out="$OUT_ROOT/$NAME/chunk_${s}_${e}"
    metrics="$out/$BENCHMARK/test_${TEMPLATE}_-1_seed${SEED}_t${TEMPERATURE}_s${s}_e${e}_metrics.json"
    if [ -f "$metrics" ]; then
        echo ">>> chunk $s-$e already done, skipping"
        continue
    fi

    start=$(date +%s)
    python -u math_eval.py \
        --model_name_or_path "$MODEL_PATH" \
        --data_dir "$DATA_DIR" \
        --data_name "$BENCHMARK" \
        --output_dir "$out" \
        --split test \
        --prompt_type "$TEMPLATE" \
        --num_test_sample -1 \
        --start "$s" --end "$e" \
        --n_sampling "$N_SAMPLING" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --max_tokens_per_call "$MAX_TOKENS" \
        --seed "$SEED" \
        --use_vllm \
        --save_outputs \
        --overwrite
    echo ">>> chunk $s-$e finished in $(( $(date +%s) - start ))s"
done

echo ">>> $NAME done. Results: $OUT_ROOT/$NAME/chunk_*/$BENCHMARK/*.jsonl"
