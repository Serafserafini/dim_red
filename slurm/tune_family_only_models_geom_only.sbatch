#!/bin/bash
#SBATCH --job-name=dimred_tune_family_only_geom_only
#SBATCH --partition=GPU-a100,GPU-a40,GPU-a100s
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2

# Round 2 of slurm/tune_family_only_models.sbatch (see
# experiments/family_only_sweep_notes.md, not tracked by git): re-runs the
# entire round 1 family-only comparison (vae/autoencoder/supcon) with
# soap.element_agnostic: true / pyxtal.n_species: 1 -- see
# configs/tuning_sweep_vae_family_only_geom_only.example.yaml's header
# comment for the full rationale. Same 3-task array structure, same grid
# axes per model kind as round 1, pointed at the new
# configs/tuning_sweep_*_family_only_geom_only.example.yaml configs and
# their own shared dataset cache (distinct dataset cache key from round 1,
# hence its own --cache-dir).
#
# REQUIRES runs/_shared_dataset_cache_family_only_geom_only to already be
# built (np.savez isn't atomic) -- run the warmup job first and chain this
# one on it:
#   WARMUP_ID=$(sbatch --parsable slurm/warmup_shared_dataset_family_only_geom_only.sbatch)
#   sbatch --dependency=afterok:${WARMUP_ID} slurm/tune_family_only_models_geom_only.sbatch
#
# To rebuild/refresh a cross-model table combining all three afterwards:
#   dimred-benchmark runs/tuning_vae_family_only_geom_only/<date>-<n> \
#       runs/tuning_autoencoder_family_only_geom_only/<date>-<n> \
#       runs/tuning_supcon_family_only_geom_only/<date>-<n> \
#       --output runs/family_only_geom_only_benchmark.csv --plot --plot-data-file true

set -euo pipefail

echo "$(hostname) - array task ${SLURM_ARRAY_TASK_ID} (family_only geom_only round 2)"

# .novalocal nodes are a known-flaky node class on this cluster (recurring
# CUDA/jaxlib and /tmp-permission failures unrelated to dim_red itself). If
# we land on one, resubmit just this array task with that node excluded and
# exit this attempt cleanly. Same guard as the other slurm/*.sbatch scripts
# in this project. Capped so a cluster with nothing but .novalocal nodes
# free still fails loudly instead of looping forever.
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
        "${SLURM_SUBMIT_DIR}/slurm/tune_family_only_models_geom_only.sbatch"
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
SWEEP_CONFIG="configs/tuning_sweep_${MODEL_KIND}_family_only_geom_only.example.yaml"

echo "Task ${SLURM_ARRAY_TASK_ID}: tuning model_kind=${MODEL_KIND} (${SWEEP_CONFIG})"
dimred-sweep "${SWEEP_CONFIG}" --cache-dir runs/_shared_dataset_cache_family_only_geom_only

SWEEP_DIR=$(ls -td "runs/tuning_${MODEL_KIND}_family_only_geom_only"/*/ | head -1)
SWEEP_DIR=${SWEEP_DIR%/}
echo "Sweep finished: ${SWEEP_DIR}"

dimred-benchmark "${SWEEP_DIR}" --output "${SWEEP_DIR}/benchmark_table.csv"
echo "Benchmark table ready: ${SWEEP_DIR}/benchmark_table.csv"
