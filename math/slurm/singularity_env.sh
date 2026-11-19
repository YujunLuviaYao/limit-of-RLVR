# Sourced by the sbatch scripts (on the host). Defines where the Singularity image,
# overlay and conda env live, and run_in_container, which runs a bash command inside
# the container with the conda env active. Override any default by exporting it
# before sbatch, e.g.  OVERLAY=/scratch/$USER/other.ext3 sbatch slurm/env_check.sbatch

export SIF="${SIF:-/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
export OVERLAY="${OVERLAY:-$SCRATCH/rlvr/overlay-25GB-500K.ext3}"
export CONDA_ROOT="${CONDA_ROOT:-/ext3/miniconda3}"
export CONDA_ENV="${CONDA_ENV:-/ext3/envs/rlvr}"

# Large files go to scratch, not home.
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export MODELS_DIR="${MODELS_DIR:-$SCRATCH/models}"

for f in "$SIF" "$OVERLAY"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found (set SIF / OVERLAY to override)" >&2
        exit 1
    fi
done

# Usage: run_in_container '<bash command>'
# The host environment (SCRATCH, HF_HOME, SLURM_*, ...) is passed into the container.
# --nv exposes the GPU; the overlay is mounted read-only so several jobs can share it.
run_in_container() {
    singularity exec --nv --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c "
        set -eo pipefail
        export PATH=\"\$CONDA_ROOT/bin:\$PATH\"
        source \"\$CONDA_ROOT/etc/profile.d/conda.sh\"
        conda activate \"\$CONDA_ENV\"
        $1"
}
