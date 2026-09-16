#!/bin/bash
#SBATCH --job-name=dimred_viztail_round8
#SBATCH --partition=GPU-a100,GPU-a40,GPU-a100s
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-3

# Trains the 4 round-8 visualization-tail combinations requested: 2 target
# bodies (round 7's overall-best supcon run excluding round 1; round 8's
# best balanced-batching supcon run) x 2 modes (family_only,
# family_and_spacegroup) -- see the 4
# configs/tail_train_visualization_round8_*.example.yaml files' own header
# comments for the full rationale. Small frozen-body tail trainings (no
# SOAP recompute, no body forward pass), so all 4 comfortably fit in one
# short array job.

set -euo pipefail

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
    echo "$(hostname) is a .novalocal node -- resubmitting array task ${SLURM_ARRAY_TASK_ID} excluding [${EXCLUDE_LIST}] (retry $((RETRY_COUNT + 1))/${MAX_RETRIES})"
    sbatch \
        --array="${SLURM_ARRAY_TASK_ID}" \
        --exclude="${EXCLUDE_LIST//:/,}" \
        --export="ALL,DIMRED_NOVALOCAL_EXCLUDE=${EXCLUDE_LIST},DIMRED_NOVALOCAL_RETRY=$((RETRY_COUNT + 1))" \
        "${SLURM_SUBMIT_DIR}/slurm/train_round8_visualization_tails.sbatch"
    exit 0
fi

source ~/miniconda3/bin/activate
conda activate dmred
if [ "${CONDA_DEFAULT_ENV:-}" != "dmred" ]; then
    echo "ERROR: dmred conda environment did not activate (got '${CONDA_DEFAULT_ENV:-unset}')" >&2
    exit 1
fi

cd "${SLURM_SUBMIT_DIR}"

TOPMODEL_RUN="runs/tuning_supcon_family_only_round7/20260914-1/model-supcon_hd-128-64_pyxtal-cub-hex-mon-ort-tet-tri-tri_nsp1_supcon-family_only_tau0.05_lf1-3"
BALANCED_RUN="runs/tuning_supcon_family_only_round8/20260915-1/model-supcon_hd-128-64_pyxtal-cub-hex-mon-ort-tet-tri-tri_nsp1_supcon-family_only_tau0.1_lf1-4"

CONFIGS=(
    "configs/tail_train_visualization_round8_topmodel_family_only.example.yaml|${TOPMODEL_RUN}"
    "configs/tail_train_visualization_round8_topmodel_family_and_spacegroup.example.yaml|${TOPMODEL_RUN}"
    "configs/tail_train_visualization_round8_supcon_balanced_family_only.example.yaml|${BALANCED_RUN}"
    "configs/tail_train_visualization_round8_supcon_balanced_family_and_spacegroup.example.yaml|${BALANCED_RUN}"
)

ENTRY="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
TAIL_CONFIG="${ENTRY%%|*}"
RUN_DIR="${ENTRY##*|}"

echo "Task ${SLURM_ARRAY_TASK_ID}: ${TAIL_CONFIG} -> ${RUN_DIR}"
dimred-train-tail "${TAIL_CONFIG}" "${RUN_DIR}"
echo "Done: ${TAIL_CONFIG}"
