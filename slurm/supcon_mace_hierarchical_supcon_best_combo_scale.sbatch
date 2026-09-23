#!/bin/bash
#SBATCH --job-name=dimred_supcon_mace_hierarchical_supcon
#SBATCH --partition=GPU-a100,GPU-a40,GPU-a100s
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00

# Per-family SupCon SG experts (real per-family body training, on the
# native MACE embedding -- see _compute_native_mace_features) for the
# already-completed model_kind: supcon_mace run
# (experiments/supcon_mace_best_combo_scale.yaml). Trains 7 independent
# SupCon SG bodies (one per family) + classifier + visualizer each, so
# this takes noticeably longer than the frozen-mace hierarchical_supcon
# job did (that one trained no per-family bodies at all).

set -euo pipefail

echo "$(hostname) - supcon_mace hierarchical_supcon (per-family SG experts)"

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
        "${SLURM_SUBMIT_DIR}/slurm/supcon_mace_hierarchical_supcon_best_combo_scale.sbatch"
    exit 0
fi

source ~/miniconda3/bin/activate
conda activate dmred
if [ "${CONDA_DEFAULT_ENV:-}" != "dmred" ]; then
    echo "ERROR: dmred conda environment did not activate (got '${CONDA_DEFAULT_ENV:-unset}')" >&2
    exit 1
fi

cd "${SLURM_SUBMIT_DIR}"

dimred-train-tail \
    configs/tail_train_hierarchical_supcon_supcon_mace_best_combo_scale.example.yaml \
    runs/model-supcon_mace_hd-256-128_pyxtal-cub-hex-mon-ort-tet-tri-tri_nsp1_supcon-family_only_tau0.05_lf1

echo "Run complete."
