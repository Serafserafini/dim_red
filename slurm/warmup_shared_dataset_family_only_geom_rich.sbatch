#!/bin/bash
#SBATCH --job-name=dimred_warmup_family_only_geom_rich
#SBATCH --partition=GPU-a100,GPU-a40,GPU-a100s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

# Round 3 of slurm/warmup_shared_dataset_family_only_geom_only.sbatch (see
# experiments/family_only_geom_only_round2_notes.md, not tracked by git):
# pre-builds, ONCE, the SOAP dataset cache that
# configs/tuning_sweep_{vae,autoencoder,supcon}_family_only_geom_rich
# .example.yaml all share (soap.r_cut/n_max/l_max raised to 6.0/8/6 to
# compensate for the 12960->40-dimensional SOAP collapse round 2 measured
# from element_agnostic: true/n_species: 1 alone -- a NEW dataset cache
# key, distinct from both round 1's runs/_shared_dataset_cache_family_only
# and round 2's runs/_shared_dataset_cache_family_only_geom_only).
#
# A richer SOAP basis (n_max=8/l_max=6 vs rounds 1/2's 4/3) costs more CPU
# time per structure than the earlier rounds' warmup -- still no --gres=gpu
# needed (dim_red.soap.compute_soap is pure dscribe/CPU work).
#
# Submit BEFORE the array job and chain the array on it (required, not just
# tidy -- np.savez isn't atomic):
#   WARMUP_ID=$(sbatch --parsable slurm/warmup_shared_dataset_family_only_geom_rich.sbatch)
#   sbatch --dependency=afterok:${WARMUP_ID} slurm/tune_family_only_models_geom_rich.sbatch

set -euo pipefail

echo "$(hostname) - dataset cache warm-up (family_only, geom_rich round 3)"

# .novalocal nodes are a known-flaky node class on this cluster (recurring
# CUDA/jaxlib and /tmp-permission failures unrelated to dim_red itself) --
# resubmit a fresh job excluding this node and exit this attempt cleanly.
# Same guard as the other slurm/*.sbatch scripts in this project. Capped so
# a cluster with nothing but .novalocal nodes free still fails loudly
# instead of looping forever.
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
        "${SLURM_SUBMIT_DIR}/slurm/warmup_shared_dataset_family_only_geom_rich.sbatch"
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
    runs/_shared_dataset_cache_family_only_geom_rich \
    configs/tuning_sweep_vae_family_only_geom_rich.example.yaml \
    configs/tuning_sweep_autoencoder_family_only_geom_rich.example.yaml \
    configs/tuning_sweep_supcon_family_only_geom_rich.example.yaml

echo "Warm-up complete: runs/_shared_dataset_cache_family_only_geom_rich"
