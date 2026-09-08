#!/bin/bash
#SBATCH --job-name=dimred_warmup_family_only
#SBATCH --partition=GPU-a100,GPU-a40,GPU-a100s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

# Pre-builds, ONCE, the SOAP dataset cache that vae/autoencoder/supcon's
# configs/tuning_sweep_*_family_only.example.yaml configs all share
# (identical pyxtal + soap + augmentation settings) -- see
# warmup_shared_dataset.py's docstring for why this matters: without it,
# tune_family_only_models.sbatch's three parallel array tasks would each
# build (and store on disk) their own copy of the same ~17500-structure
# SOAP embeddings, tripling both compute time and disk usage for identical
# data. This is a family_only-round sibling of warmup_shared_dataset.sbatch,
# using its own cache dir (runs/_shared_dataset_cache_family_only) so it
# never collides with the production family_and_spacegroup sweeps' own
# runs/_shared_dataset_cache.
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
#   WARMUP_ID=$(sbatch --parsable slurm/warmup_shared_dataset_family_only.sbatch)
#   sbatch --dependency=afterok:${WARMUP_ID} slurm/tune_family_only_models.sbatch

set -euo pipefail

echo "$(hostname) - dataset cache warm-up (family_only)"

# .novalocal nodes are a known-flaky node class on this cluster (recurring
# CUDA/jaxlib and /tmp-permission failures unrelated to dim_red itself) --
# resubmit a fresh job excluding this node and exit this attempt cleanly.
# Same guard as warmup_shared_dataset.sbatch/tune_all_models.sbatch (see
# those files for why this can't be `scontrol update ExcludeNodes` +
# `scontrol requeue` on the current job -- it's already running by the
# time we detect the bad node, and this cluster refuses ExcludeNodes
# updates on running jobs). Capped so a cluster with nothing but
# .novalocal nodes free still fails loudly instead of looping forever.
if [[ "$(hostname)" == *.novalocal ]]; then
    MAX_RETRIES=5
    RETRY_COUNT="${DIMRED_NOVALOCAL_RETRY:-0}"
    if [ "${RETRY_COUNT}" -ge "${MAX_RETRIES}" ]; then
        echo "ERROR: still landing on .novalocal nodes after ${RETRY_COUNT} retries, giving up on $(hostname)" >&2
        exit 1
    fi
    EXCLUDE_LIST="$(hostname -s)"
    if [ -n "${DIMRED_NOVALOCAL_EXCLUDE:-}" ]; then
        EXCLUDE_LIST="${DIMRED_NOVALOCAL_EXCLUDE}:$(hostname -s)"
    fi
    echo "$(hostname) is a .novalocal node -- resubmitting excluding [${EXCLUDE_LIST}] (retry $((RETRY_COUNT + 1))/${MAX_RETRIES})"
    sbatch \
        --exclude="${EXCLUDE_LIST//:/,}" \
        --export="ALL,DIMRED_NOVALOCAL_EXCLUDE=${EXCLUDE_LIST},DIMRED_NOVALOCAL_RETRY=$((RETRY_COUNT + 1))" \
        "${SLURM_SUBMIT_DIR}/slurm/warmup_shared_dataset_family_only.sbatch"
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
    runs/_shared_dataset_cache_family_only \
    configs/tuning_sweep_vae_family_only.example.yaml \
    configs/tuning_sweep_autoencoder_family_only.example.yaml \
    configs/tuning_sweep_supcon_family_only.example.yaml

echo "Warm-up complete: runs/_shared_dataset_cache_family_only"
