#!/bin/bash
#SBATCH --job-name=dimred_supcon_mace_best_combo_scale
#SBATCH --partition=GPU-a100,GPU-a40,GPU-a100s
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

# Full round-15-scale test of model_kind: supcon_mace -- best_combo's own
# SupCon body/loss trained on frozen MACE-MP-0 features instead of SOAP.
# Same data quantity as best_combo and the model_kind: mace comparison
# (500/family, 17500 structures). Unlike the frozen mace/hierarchical_supcon
# jobs, this one actually trains a body (SupCon contrastive loss over up to
# 200 epochs with early stopping), so it's a real training job, not just
# a frozen forward pass + tiny tail.

set -euo pipefail

echo "$(hostname) - supcon_mace best_combo-scale test"

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
        "${SLURM_SUBMIT_DIR}/slurm/supcon_mace_best_combo_scale.sbatch"
    exit 0
fi

source ~/miniconda3/bin/activate
conda activate dmred
if [ "${CONDA_DEFAULT_ENV:-}" != "dmred" ]; then
    echo "ERROR: dmred conda environment did not activate (got '${CONDA_DEFAULT_ENV:-unset}')" >&2
    exit 1
fi

cd "${SLURM_SUBMIT_DIR}"

dimred-run experiments/supcon_mace_best_combo_scale.yaml

echo "Run complete."
