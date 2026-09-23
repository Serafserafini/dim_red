#!/bin/bash
#SBATCH --job-name=dimred_mace_family_best_combo_scale
#SBATCH --partition=GPU-a100,GPU-a40,GPU-a100s
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

# Full round-15-scale test of model_kind: mace (frozen MACE-MP-0 embedding)
# as a drop-in replacement for the SupCon encoder+projection body -- same
# data quantity as best_combo (500/family, 17500 structures after
# augmentation), not the reduced 100/family first test
# (mace_family_first_test.yaml). Chains the family body+tails run and the
# hierarchical_supcon (family -> per-family SG classifier+visualizer) tail
# training in one job, capturing the run_dir dimred-run prints ("Run
# complete: <dir>") since its exact name depends on how many prior runs
# already exist under runs/ with the same config hash prefix.

set -euo pipefail

echo "$(hostname) - mace family best_combo-scale test"

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
        "${SLURM_SUBMIT_DIR}/slurm/mace_family_best_combo_scale.sbatch"
    exit 0
fi

source ~/miniconda3/bin/activate
conda activate dmred
if [ "${CONDA_DEFAULT_ENV:-}" != "dmred" ]; then
    echo "ERROR: dmred conda environment did not activate (got '${CONDA_DEFAULT_ENV:-unset}')" >&2
    exit 1
fi

cd "${SLURM_SUBMIT_DIR}"

RUN_OUTPUT=$(dimred-run experiments/mace_family_best_combo_scale.yaml | tee /dev/stderr)
RUN_DIR=$(echo "${RUN_OUTPUT}" | grep -oP '(?<=^Run complete: ).*')
if [ -z "${RUN_DIR}" ]; then
    echo "ERROR: could not parse run_dir from dimred-run output" >&2
    exit 1
fi
echo "Family run complete: ${RUN_DIR}"

dimred-train-tail \
    configs/tail_train_hierarchical_supcon_mace_best_combo_scale.example.yaml \
    "${RUN_DIR}"

echo "Run complete."
