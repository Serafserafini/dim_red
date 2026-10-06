#!/bin/bash
#SBATCH --job-name=dimred_round19_phase1
#SBATCH --partition=GPU-a100,GPU-a40,GPU-a100s
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# Generic round19 hidden-dim-sweep phase-1 runner (model_kind: supcon_mace,
# body + auto-attached classification/visualization tails). Pass the config
# path via CONFIG_YAML (--export=CONFIG_YAML=...), e.g.:
#
#   sbatch --job-name=dimred_round19_baseline \
#       --export=ALL,CONFIG_YAML=experiments/round19_sweep/baseline.yaml \
#       slurm/round19_phase1.sbatch
#
# All 9 round19 sweep configs share the exact same pyxtal/augmentation/mace
# fields, so they share one dataset-cache entry (runs/_dataset_cache) --
# submit the baseline config first and wait for it to complete (it pays the
# one-time dataset-build cost), then the other 8 configs' dataset builds
# become cache hits.

set -euo pipefail

: "${CONFIG_YAML:?CONFIG_YAML must be set via --export}"

echo "$(hostname) - round19 phase1: ${CONFIG_YAML}"

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
        --job-name="${SLURM_JOB_NAME}" \
        --exclude="${EXCLUDE_LIST//:/,}" \
        --export="ALL,CONFIG_YAML=${CONFIG_YAML},DIMRED_NOVALOCAL_EXCLUDE=${EXCLUDE_LIST},DIMRED_NOVALOCAL_RETRY=$((RETRY_COUNT + 1))" \
        "${SLURM_SUBMIT_DIR}/slurm/round19_phase1.sbatch"
    exit 0
fi

source ~/miniconda3/bin/activate
conda activate dmred
if [ "${CONDA_DEFAULT_ENV:-}" != "dmred" ]; then
    echo "ERROR: dmred conda environment did not activate (got '${CONDA_DEFAULT_ENV:-unset}')" >&2
    exit 1
fi

cd "${SLURM_SUBMIT_DIR}"

dimred-run "${CONFIG_YAML}"

echo "Run complete."
