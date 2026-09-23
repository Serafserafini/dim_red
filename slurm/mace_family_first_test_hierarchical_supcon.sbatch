#!/bin/bash
#SBATCH --job-name=dimred_mace_hierarchical_supcon_first_test
#SBATCH --partition=GPU-a100,GPU-a40,GPU-a100s
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

# Trains the hierarchical_supcon (family -> per-family classifier+visualizer,
# frozen MACE embedding standing in for the per-family SupCon SG
# encoder+projection) tail on top of the already-completed
# mace_family_first_test.yaml run -- see
# configs/tail_train_hierarchical_supcon_mace_family_first_test.example.yaml.
# No new MACE forward pass: train_tail reads straight from that run's
# cached embeddings.npz.

set -euo pipefail

echo "$(hostname) - mace hierarchical_supcon (family -> SG classifier+viz) first test"

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
        "${SLURM_SUBMIT_DIR}/slurm/mace_family_first_test_hierarchical_supcon.sbatch"
    exit 0
fi

source ~/miniconda3/bin/activate
conda activate dmred
if [ "${CONDA_DEFAULT_ENV:-}" != "dmred" ]; then
    echo "ERROR: dmred conda environment did not activate (got '${CONDA_DEFAULT_ENV:-unset}')" >&2
    exit 1
fi

cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"

dimred-train-tail \
    configs/tail_train_hierarchical_supcon_mace_family_first_test.example.yaml \
    runs/model-mace_hd-1_pyxtal-cub-hex-mon-ort-tet-tri-tri_nsp1-2

echo "Run complete."
