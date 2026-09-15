#!/bin/bash
#SBATCH --job-name=dimred_warmup_family_only_round8
#SBATCH --partition=GPU-a100,GPU-a40,GPU-a100s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

# Round 8 ("spacegroup_balanced") of slurm/warmup_shared_dataset_family_only_round7
# -- but round 7 itself never needed its own warmup (it reused round 3's
# cache directly). This warmup pre-builds, ONCE, the SOAP dataset cache
# that configs/tuning_sweep_{vae,autoencoder,supcon}_family_only_round8
# .example.yaml all share: same soap/train settings as round 7
# (element_agnostic: true, n_species: 1, dim=252), but
# `pyxtal.structures_per_family: 500`/`distribution: uniform` swapped for
# `structures_per_spacegroup: 15` (15 structures for EVERY one of the 230
# space groups spanned by the 7 families, instead of 500 per family split
# unevenly across each family's own space groups) -- a NEW dataset cache
# key, distinct from every earlier round. See
# experiments/family_only_round8_notes.md for the resulting per-family
# raw-structure imbalance (an explicit, expected consequence of sampling
# uniformly per space group instead of per family).
#
# Same SOAP richness (r_cut/n_max/l_max: 6.0/8/6) as round 3/5/7 -- similar
# per-structure CPU cost, comparable total structure count (15*230=3450
# raw vs round 7's 500*7=3500), so a similar warmup wall-clock budget.
#
# Submit BEFORE the array job and chain the array on it (required, not just
# tidy -- np.savez isn't atomic):
#   WARMUP_ID=$(sbatch --parsable slurm/warmup_shared_dataset_family_only_round8.sbatch)
#   sbatch --dependency=afterok:${WARMUP_ID} slurm/tune_family_only_models_round8.sbatch

set -euo pipefail

echo "$(hostname) - dataset cache warm-up (family_only, round 8)"

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
        "${SLURM_SUBMIT_DIR}/slurm/warmup_shared_dataset_family_only_round8.sbatch"
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
    runs/_shared_dataset_cache_family_only_round8 \
    configs/tuning_sweep_vae_family_only_round8.example.yaml \
    configs/tuning_sweep_autoencoder_family_only_round8.example.yaml \
    configs/tuning_sweep_supcon_family_only_round8.example.yaml

echo "Warm-up complete: runs/_shared_dataset_cache_family_only_round8"
