#!/bin/bash
#SBATCH --job-name=dimred_tune_family_only_round12
#SBATCH --partition=GPU-a100,GPU-a40,GPU-a100s
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2

# Round 12 ("scale_invariance_richer_cutoff") of
# slurm/tune_family_only_models_round10: re-runs the family-only
# comparison with a richer relative cutoff (r_cut=5.0/sigma=0.417
# instead of round 10's 3.0/0.25), against a NEW dataset cache
# (runs/_shared_dataset_cache_family_only_round12). REQUIRES that warmup
# to have completed first (chain via --dependency=afterok). Same 3-task
# array structure as every earlier round.
#
# To rebuild/refresh a cross-model table combining all three afterwards:
#   dimred-benchmark runs/tuning_vae_family_only_round12/<date>-<n> \
#       runs/tuning_autoencoder_family_only_round12/<date>-<n> \
#       runs/tuning_supcon_family_only_round12/<date>-<n> \
#       --output runs/family_only_round12_benchmark.csv --plot --plot-data-file true

set -euo pipefail

echo "$(hostname) - array task ${SLURM_ARRAY_TASK_ID} (family_only round 12, scale_invariance_richer_cutoff)"

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
        "${SLURM_SUBMIT_DIR}/slurm/tune_family_only_models_round12.sbatch"
    exit 0
fi

source ~/miniconda3/bin/activate
conda activate dmred
if [ "${CONDA_DEFAULT_ENV:-}" != "dmred" ]; then
    echo "ERROR: dmred conda environment did not activate (got '${CONDA_DEFAULT_ENV:-unset}')" >&2
    exit 1
fi

cd "${SLURM_SUBMIT_DIR}"

MODEL_KINDS=(vae autoencoder supcon)
MODEL_KIND="${MODEL_KINDS[$SLURM_ARRAY_TASK_ID]}"
SWEEP_CONFIG="configs/tuning_sweep_${MODEL_KIND}_family_only_round12.example.yaml"

echo "Task ${SLURM_ARRAY_TASK_ID}: tuning model_kind=${MODEL_KIND} (${SWEEP_CONFIG})"
dimred-sweep "${SWEEP_CONFIG}" --cache-dir runs/_shared_dataset_cache_family_only_round12

SWEEP_DIR=$(ls -td "runs/tuning_${MODEL_KIND}_family_only_round12"/*/ | head -1)
SWEEP_DIR=${SWEEP_DIR%/}
echo "Sweep finished: ${SWEEP_DIR}"

dimred-benchmark "${SWEEP_DIR}" --output "${SWEEP_DIR}/benchmark_table.csv"
echo "Benchmark table ready: ${SWEEP_DIR}/benchmark_table.csv"
