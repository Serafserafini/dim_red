#!/bin/bash
#SBATCH --job-name=dimred_round19_phase2
#SBATCH --partition=GPU-a100,GPU-a40,GPU-a100s
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00

# Generic round19 hidden-dim-sweep phase-2 runner (tail_kind:
# hierarchical_supcon, oracle-routing SG experts). Pass the tail config path
# via TAIL_CONFIG_YAML and the phase-1 run directory via RUN_DIR
# (--export=ALL,TAIL_CONFIG_YAML=...,RUN_DIR=...), e.g.:
#
#   sbatch --job-name=dimred_round19_baseline_p2 \
#       --export=ALL,TAIL_CONFIG_YAML=configs/round19_sweep/tail_train_baseline.example.yaml,RUN_DIR=runs/... \
#       slurm/round19_phase2.sbatch

set -euo pipefail

: "${TAIL_CONFIG_YAML:?TAIL_CONFIG_YAML must be set via --export}"
: "${RUN_DIR:?RUN_DIR must be set via --export}"

echo "$(hostname) - round19 phase2: ${TAIL_CONFIG_YAML} on ${RUN_DIR}"

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
        --export="ALL,TAIL_CONFIG_YAML=${TAIL_CONFIG_YAML},RUN_DIR=${RUN_DIR},DIMRED_NOVALOCAL_EXCLUDE=${EXCLUDE_LIST},DIMRED_NOVALOCAL_RETRY=$((RETRY_COUNT + 1))" \
        "${SLURM_SUBMIT_DIR}/slurm/round19_phase2.sbatch"
    exit 0
fi

source ~/miniconda3/bin/activate
conda activate dmred
if [ "${CONDA_DEFAULT_ENV:-}" != "dmred" ]; then
    echo "ERROR: dmred conda environment did not activate (got '${CONDA_DEFAULT_ENV:-unset}')" >&2
    exit 1
fi

cd "${SLURM_SUBMIT_DIR}"

dimred-train-tail "${TAIL_CONFIG_YAML}" "${RUN_DIR}"

echo "Run complete."
