#!/bin/bash
# Phase 2 analysis on CPU (no GPU needed): RQ1 summary, revised grader + corrected pass@k
# (RQ2/RQ3), code execution (RQ4). Submit from the math/ directory:
#   cd ~/limit-of-RLVR/math && sbatch slurm/phase2_analysis.sbatch
# Everything is written to $SCRATCH/rlvr/phase2 (override with P2_OUT); the small reports
# (*.md, *.csv, *.png) are what to look at / send back.
# If the partition refuses a job without a GPU, uncomment the --gres line.
#SBATCH --job-name=rlvr-phase2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --account=ds_ga_1011-2026fa
#SBATCH --partition=c12m85-a100-1
##SBATCH --gres=gpu:1

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/singularity_env.sh
export P2_OUT="${P2_OUT:-$SCRATCH/rlvr/phase2}"
export ANN="${ANN:-$SCRATCH/rlvr/annotation_b1}"
export WORKERS="${SLURM_CPUS_PER_TASK:-12}"
mkdir -p "$P2_OUT"
echo "Host: $(hostname)  out: $P2_OUT  workers: $WORKERS"

run_in_container '
    set -x
    B="$SCRATCH/rlvr/main_out*/Qwen2.5-7B/chunk_*/math500/*.jsonl"
    R="$SCRATCH/rlvr/main_out*/SimpleRL-Zoo-7B/chunk_*/math500/*.jsonl"

    # RQ1: label summary (batch 1 was drawn from the 200-problem results in main_out/)
    python summarize_annotation.py --labels annotation/labels/b1_final.csv --key "$ANN/key.csv" \
        --raw annotation/labels/b1_yujun.csv annotation/labels/b1_ganlin.csv annotation/labels/b1_haotong.csv \
        --out "$P2_OUT/rq1_summary.md" \
        "Qwen2.5-7B=$SCRATCH/rlvr/main_out/Qwen2.5-7B/chunk_*/math500/*.jsonl" \
        "SimpleRL-Zoo-7B=$SCRATCH/rlvr/main_out/SimpleRL-Zoo-7B/chunk_*/math500/*.jsonl"

    # RQ2/RQ3: validate the revised grader on the labels, then re-grade all 500 problems
    python regrade.py --validate annotation/labels/b1_final.csv --items "$ANN/items_full.jsonl" \
        --key "$ANN/key.csv" --report "$P2_OUT/grader_validation.md"
    python regrade.py "Qwen2.5-7B=$B" "SimpleRL-Zoo-7B=$R" --out_dir "$P2_OUT/regraded" --workers "$WORKERS"
    python passk_minimal.py --bootstrap 2000 --quiet --csv "$P2_OUT/passk_original.csv" \
        "Base=$B" "RL=$R" > "$P2_OUT/passk_original.txt"
    python passk_minimal.py --bootstrap 2000 --quiet --csv "$P2_OUT/passk_regraded.csv" \
        --plot "$P2_OUT/fig_passk_regraded.png" --title "MATH500, revised grader" \
        "Base (Qwen2.5-7B)=$P2_OUT/regraded/Qwen2.5-7B/math500/*.jsonl" \
        "RL (SimpleRL-Zoo-7B)=$P2_OUT/regraded/SimpleRL-Zoo-7B/math500/*.jsonl" > "$P2_OUT/passk_regraded.txt"

    # RQ4: run the code in the responses
    python run_code.py --items "$ANN/items_full.jsonl" --labels annotation/labels/b1_final.csv \
        --key "$ANN/key.csv" --workers "$WORKERS" --report "$P2_OUT/code_on_labels.md"
    python run_code.py "Qwen2.5-7B=$P2_OUT/regraded/Qwen2.5-7B/math500/*.jsonl" \
        "SimpleRL-Zoo-7B=$P2_OUT/regraded/SimpleRL-Zoo-7B/math500/*.jsonl" \
        --out_dir "$P2_OUT/code_exec" --workers "$WORKERS"
    python passk_minimal.py --bootstrap 2000 --quiet --csv "$P2_OUT/passk_tool.csv" \
        --plot "$P2_OUT/fig_passk_tool.png" --title "MATH500, revised grader + code run" \
        "Base (Qwen2.5-7B)=$P2_OUT/code_exec/tool/Qwen2.5-7B/math500/*.jsonl" \
        "RL (SimpleRL-Zoo-7B)=$P2_OUT/code_exec/tool/SimpleRL-Zoo-7B/math500/*.jsonl" > "$P2_OUT/passk_tool.txt"

    cp "$P2_OUT/regraded/regrade_summary.md" "$P2_OUT/code_exec/code_summary.md" "$P2_OUT/" 2>/dev/null || true
    ls -la "$P2_OUT"
'
