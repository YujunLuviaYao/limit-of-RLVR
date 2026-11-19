# Main experiment on a Slurm cluster

Base vs. RL model (Qwen2.5-7B vs. SimpleRL-Zoo-7B), 200 MATH500 problems sampled in
proportion to the five difficulty levels, n = 128 samples per problem, temperature 0.6,
top-p 0.95, up to 16k tokens (the paper's MATH500 setup).

All commands run from `limit-of-RLVR/math`.

1. **Environment (once, on a login node).**
   ```bash
   module load anaconda3        # or however your cluster provides conda
   bash setup_env.sh            # creates $SCRATCH/envs/rlvr
   conda activate $SCRATCH/envs/rlvr
   ```
   Then check it on a GPU node (fill in account/partition in the file first):
   `sbatch slurm/env_check.sbatch` runs a 2-problem smoke test with Qwen2.5-0.5B;
   the log `slurm/rlvr-envcheck_<job>.out` should end with pass@k numbers.
2. **Subset (already committed).** `subsets/math500_strat200_seed0/` was made with
   `python make_subset.py`; regenerate only to change size or seed.
3. **Models (on a login node if compute nodes have no internet).**
   ```bash
   huggingface-cli download Qwen/Qwen2.5-7B --local-dir $SCRATCH/models/Qwen2.5-7B
   huggingface-cli download hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo --local-dir $SCRATCH/models/SimpleRL-Zoo-7B
   ```
4. **Submit.** Fill in the account/partition lines in `slurm/main_experiment.sbatch`, then
   ```bash
   sbatch slurm/main_experiment.sbatch
   ```
   One array task per model, one GPU each (>= 24 GB). Logs: `slurm/rlvr-main_<job>_<task>.out`.
   If a task hits the time limit, resubmit the same command: finished chunks are skipped.
5. **pass@k.**
   ```bash
   python passk_minimal.py --plot main_out/passk.png \
       'Qwen2.5-7B=main_out/Qwen2.5-7B/chunk_*/math500/*.jsonl' \
       'SimpleRL-Zoo-7B=main_out/SimpleRL-Zoo-7B/chunk_*/math500/*.jsonl'
   ```

`run_main.sh` sets `KEEP_MODEL_DIR=1`, so `math_eval.py` no longer deletes the model
directory it loads (the original pipeline deletes it because it evaluates a throwaway copy).
