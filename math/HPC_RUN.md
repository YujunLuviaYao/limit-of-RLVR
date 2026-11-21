# Main experiment on NYU HPC (Slurm + Singularity)

Base vs. RL model (Qwen2.5-7B vs. SimpleRL-Zoo-7B), 200 MATH500 problems sampled in
proportion to the five difficulty levels, n = 128 samples per problem, temperature 0.6,
top-p 0.95, up to 16k tokens (the paper's MATH500 setup).

Jobs run inside a Singularity container; the Python environment lives in an ext3 overlay.
The code can stay in `/home`, but everything large goes to `$SCRATCH`:

| What | Default location | Override with |
|---|---|---|
| Container image | `/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif` | `SIF` |
| Overlay | `$SCRATCH/rlvr/overlay-25GB-500K.ext3` | `OVERLAY` |
| Miniconda (in overlay) | `/ext3/miniconda3` | `CONDA_ROOT` |
| Conda env (in overlay) | `/ext3/envs/rlvr` | `CONDA_ENV` |
| Models | `$SCRATCH/models` | `MODELS_DIR` |
| HF cache | `$SCRATCH/hf_cache` | `HF_HOME` |
| Main results | `$SCRATCH/rlvr/main_out` | `OUT_ROOT` |
| Env-check results | `$SCRATCH/rlvr/envcheck_out` | `OUT_ROOT` |

Defaults are set in `slurm/singularity_env.sh`; export a variable before `sbatch` to change it.

## 1. One-time environment setup (login node)

**Copy an overlay** (25 GB, 500K files) to scratch and unzip it. The public overlay
templates live under `/scratch/work/public/overlay-fs-ext3/` (see the NYU HPC docs if the
name has changed):
```bash
mkdir -p $SCRATCH/rlvr
cp /scratch/work/public/overlay-fs-ext3/overlay-25GB-500K.ext3.gz $SCRATCH/rlvr/
gunzip $SCRATCH/rlvr/overlay-25GB-500K.ext3.gz
```

**Enter the container with the overlay writable (`:rw`)** so you can install into it.
Only one process may have the overlay open read-write, so close any other session using it:
```bash
singularity exec --overlay $SCRATCH/rlvr/overlay-25GB-500K.ext3:rw \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash
```

**Inside the container**, install Miniconda into the overlay:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p /ext3/miniconda3
```

**Still inside the container**, put conda on PATH (sourcing `conda.sh` alone is not
enough for `setup_env.sh`, which runs as a child process) and create the env:
```bash
export PATH=/ext3/miniconda3/bin:$PATH
cd ~/limit-of-RLVR/math
ENV_PREFIX=/ext3/envs/rlvr bash setup_env.sh
exit
```
`setup_env.sh` installs vLLM 0.6.3 (with torch 2.4), transformers<4.48, math-verify,
latex2sympy and the other evaluation dependencies, and prints the vLLM/torch versions.

## 2. Slurm settings

Fill in `--account` and `--partition` in `slurm/env_check.sbatch` and
`slurm/main_experiment.sbatch` (currently commented placeholders: remove the extra `#`).

## 3. Check the environment on a GPU node

Compute nodes may have no internet, so download the test model on a login node first:
```bash
python download_model.py Qwen/Qwen2.5-0.5B $SCRATCH/models/Qwen2.5-0.5B
```
(Run it inside the container with the env active, from `~/limit-of-RLVR/math`; it uses
`huggingface_hub.snapshot_download`, which works whatever `huggingface_hub` version is
installed.) Then, from `~/limit-of-RLVR/math`:
```bash
sbatch slurm/env_check.sbatch
```
It runs a 2-problem, n=2 smoke test with Qwen2.5-0.5B inside the container (overlay mounted
read-only, `--nv` for the GPU). The log `slurm/rlvr-envcheck_<job>.out` should show
`cuda ok: True` and end with pass@k numbers.

## 4. Main experiment

Download the two 7B models (login node, inside the container as in step 3):
```bash
python download_model.py Qwen/Qwen2.5-7B $SCRATCH/models/Qwen2.5-7B
python download_model.py hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo $SCRATCH/models/SimpleRL-Zoo-7B
```
Submit:
```bash
sbatch slurm/main_experiment.sbatch
```
One array task per model, one GPU each (>= 24 GB), both sharing the read-only overlay.
Logs: `slurm/rlvr-main_<job>_<task>.out`. If a task hits the time limit, resubmit the same
command: finished chunks (50 problems each) are skipped. A shorter `--time` (e.g. 3 h)
may get scheduled sooner, since the run resumes anyway.

## 5. pass@k

```bash
singularity exec --overlay $SCRATCH/rlvr/overlay-25GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -c '
    export PATH=/ext3/miniconda3/bin:$PATH && source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate /ext3/envs/rlvr && cd ~/limit-of-RLVR/math
    python passk_minimal.py --plot $SCRATCH/rlvr/main_out/passk.png \
        "Qwen2.5-7B=$SCRATCH/rlvr/main_out/Qwen2.5-7B/chunk_*/math500/*.jsonl" \
        "SimpleRL-Zoo-7B=$SCRATCH/rlvr/main_out/SimpleRL-Zoo-7B/chunk_*/math500/*.jsonl"'
```

## 6. The remaining 300 problems (full MATH500)

`subsets/math500_rest300/` holds every MATH500 problem that is not in the 200-problem subset
(made with `python make_subset.py --rest_of subsets/math500_strat200_seed0`). Fill in the
account/partition lines in `slurm/main_experiment_rest300.sbatch` as in step 2, then:
```bash
sbatch slurm/main_experiment_rest300.sbatch
```
Results go to `$SCRATCH/rlvr/main_out_rest300` (separate folder, because chunk folders are
numbered within each subset). Problems keep their MATH500 `idx`, so pass@k over all 500 problems
just globs both folders:
```bash
python passk_minimal.py --plot $SCRATCH/rlvr/passk_all500.png \
    "Qwen2.5-7B=$SCRATCH/rlvr/main_out*/Qwen2.5-7B/chunk_*/math500/*.jsonl" \
    "SimpleRL-Zoo-7B=$SCRATCH/rlvr/main_out*/SimpleRL-Zoo-7B/chunk_*/math500/*.jsonl"
```
(run inside the container as in step 5).

## Notes

- `run_main.sh` sets `KEEP_MODEL_DIR=1`, so `math_eval.py` does not delete the model
  directory it loads (the original pipeline deletes it because it evaluates a throwaway copy).
- `run_minimal.sh` (used by the env check) copies the model to `/tmp` on the compute node
  first, so the original download is never deleted.
