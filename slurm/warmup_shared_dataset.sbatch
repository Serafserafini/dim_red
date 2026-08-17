#!/bin/bash
#SBATCH --job-name=dimred_warmup
#SBATCH --partition=GPU-a100,GPU-a40,GPU-a100s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

# Pre-builds, ONCE, the SOAP dataset cache that vae/autoencoder/supcon's
# slurm/tuning_sweep_*.yaml configs all share (identical pyxtal + soap +
# augmentation settings) -- see warmup_shared_dataset.py's docstring for why
# this matters: without it, tune_all_models.sbatch's three parallel array
# tasks each build (and store on disk) their own copy of the same
# ~35000-structure SOAP embeddings, tripling both compute time and disk
# usage for identical data. This is the single biggest lever on cluster
# quota usage for that job.
#
# No --gres=gpu here on purpose: dim_red.soap.compute_soap is pure
# dscribe/CPU work and never touches jax/GPU. If this cluster has a
# CPU-only partition, point --partition at that instead of the GPU ones
# above -- running this on a GPU node still "works" but burns GPU-hour
# quota on work that doesn't need a GPU at all.
#
# Submit BEFORE the array job and chain the array on it (running this
# first, not concurrently with the array, is required, not just tidy:
# np.savez isn't atomic, so a task in the array reading the cache file
# while this job is still mid-write would see a corrupt/incomplete file):
#   WARMUP_ID=$(sbatch --parsable slurm/warmup_shared_dataset.sbatch)
#   sbatch --dependency=afterok:${WARMUP_ID} slurm/tune_all_models.sbatch

set -euo pipefail

echo "$(hostname) - dataset cache warm-up"

# .novalocal nodes are a known-flaky node class on this cluster (recurring
# CUDA/jaxlib and /tmp-permission failures unrelated to dim_red itself) --
# same guard as tune_all_models.sbatch, see that file for the full
# explanation. Capped via SLURM_RESTART_COUNT so a cluster with nothing but
# .novalocal nodes free still fails loudly instead of looping forever.
if [[ "$(hostname)" == *.novalocal ]]; then
    MAX_RETRIES=5
    RESTART_COUNT="${SLURM_RESTART_COUNT:-0}"
    if [ "${RESTART_COUNT}" -ge "${MAX_RETRIES}" ]; then
        echo "ERROR: still landing on .novalocal nodes after ${RESTART_COUNT} retries, giving up on $(hostname)" >&2
        exit 1
    fi
    echo "$(hostname) is a .novalocal node -- excluding it and requeuing (retry ${RESTART_COUNT}/${MAX_RETRIES})"
    EXISTING_EXCLUDES=$(scontrol show job -o "${SLURM_JOB_ID}" | grep -oP 'ExcludeNodes=\K\S+' || true)
    if [ -z "${EXISTING_EXCLUDES}" ] || [ "${EXISTING_EXCLUDES}" = "(null)" ]; then
        NEW_EXCLUDES="$(hostname -s)"
    else
        NEW_EXCLUDES="${EXISTING_EXCLUDES},$(hostname -s)"
    fi
    scontrol update jobid="${SLURM_JOB_ID}" ExcludeNodes="${NEW_EXCLUDES}"
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
fi

source ~/miniconda3/bin/activate
conda activate dmred
if [ "${CONDA_DEFAULT_ENV:-}" != "dmred" ]; then
    echo "ERROR: dmred conda environment did not activate (got '${CONDA_DEFAULT_ENV:-unset}')" >&2
    exit 1
fi

cd "${SLURM_SUBMIT_DIR}"

python slurm/warmup_shared_dataset.py \
    runs/_shared_dataset_cache \
    slurm/tuning_sweep_vae.yaml \
    slurm/tuning_sweep_autoencoder.yaml \
    slurm/tuning_sweep_supcon.yaml

echo "Warm-up complete: runs/_shared_dataset_cache"
