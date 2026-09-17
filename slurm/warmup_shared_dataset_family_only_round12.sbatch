#!/bin/bash
#SBATCH --job-name=dimred_warmup_family_only_round12
#SBATCH --partition=GPU-a100,GPU-a40,GPU-a100s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

# Round 12 ("scale_invariance_richer_cutoff") of
# slurm/warmup_shared_dataset_family_only_round10 -- see
# configs/tuning_sweep_vae_family_only_round12.example.yaml's header
# comment for the full rationale: richer relative cutoff (r_cut=5.0,
# sigma=0.417, same 12:1 ratio as round 10's 3.0/0.25) to test whether
# round 10's residual gap vs round 7 (concentrated in the six
# non-cubic families, per the round 10 confusion-matrix analysis) is a
# richness/r_cut issue rather than an inherent cost of scale invariance.
# New dataset cache, own warmup.
#
# Submit BEFORE the array job and chain the array on it (required, not just
# tidy -- np.savez isn't atomic):
#   WARMUP_ID=$(sbatch --parsable slurm/warmup_shared_dataset_family_only_round12.sbatch)
#   sbatch --dependency=afterok:${WARMUP_ID} slurm/tune_family_only_models_round12.sbatch

set -euo pipefail

echo "$(hostname) - dataset cache warm-up (family_only, round 12)"

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
        "${SLURM_SUBMIT_DIR}/slurm/warmup_shared_dataset_family_only_round12.sbatch"
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
    runs/_shared_dataset_cache_family_only_round12 \
    configs/tuning_sweep_vae_family_only_round12.example.yaml \
    configs/tuning_sweep_autoencoder_family_only_round12.example.yaml \
    configs/tuning_sweep_supcon_family_only_round12.example.yaml

echo "Warm-up complete: runs/_shared_dataset_cache_family_only_round12"
