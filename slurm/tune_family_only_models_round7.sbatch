#!/bin/bash
#SBATCH --job-name=dimred_tune_family_only_round7
#SBATCH --partition=GPU-a100,GPU-a40,GPU-a100s
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2

# Round 7 ("training_budget") of slurm/tune_family_only_models_round3
# (renamed from _geom_rich): re-runs the family-only comparison (vae/
# autoencoder/supcon) with the EXACT same dataset/SOAP as round 3
# (structures_per_family: 500, r_cut/n_max/l_max: 6.0/8/6 -- SAME dataset
# cache key as round 3, verified identical) but train.batch_size raised
# 32 -> 128 and train.epochs raised 100 -> 200 (matching the recovered old
# sweep's training budget; early stopping stays enabled, unlike the old
# sweep, but with a larger patience of 20 so it doesn't cut the longer run
# short). Since the dataset/SOAP fragment is unchanged from round 3, this
# reuses round 3's ALREADY-BUILT
# runs/_shared_dataset_cache_family_only_geom_rich directly -- NO new
# warmup job needed, unlike every other round in this project. Same
# 3-task array structure as every earlier round.
#
# REQUIRES runs/_shared_dataset_cache_family_only_geom_rich to already
# exist (it does -- built for round 3). If it's ever missing, run
#   sbatch slurm/warmup_shared_dataset_family_only_geom_rich.sbatch
# first.
#
# To rebuild/refresh a cross-model table combining all three afterwards:
#   dimred-benchmark runs/tuning_vae_family_only_round7/<date>-<n> \
#       runs/tuning_autoencoder_family_only_round7/<date>-<n> \
#       runs/tuning_supcon_family_only_round7/<date>-<n> \
#       --output runs/family_only_round7_benchmark.csv --plot --plot-data-file true

set -euo pipefail

echo "$(hostname) - array task ${SLURM_ARRAY_TASK_ID} (family_only round 7, training_budget)"

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
        "${SLURM_SUBMIT_DIR}/slurm/tune_family_only_models_round7.sbatch"
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
SWEEP_CONFIG="configs/tuning_sweep_${MODEL_KIND}_family_only_round7.example.yaml"

echo "Task ${SLURM_ARRAY_TASK_ID}: tuning model_kind=${MODEL_KIND} (${SWEEP_CONFIG})"
dimred-sweep "${SWEEP_CONFIG}" --cache-dir runs/_shared_dataset_cache_family_only_geom_rich

SWEEP_DIR=$(ls -td "runs/tuning_${MODEL_KIND}_family_only_round7"/*/ | head -1)
SWEEP_DIR=${SWEEP_DIR%/}
echo "Sweep finished: ${SWEEP_DIR}"

dimred-benchmark "${SWEEP_DIR}" --output "${SWEEP_DIR}/benchmark_table.csv"
echo "Benchmark table ready: ${SWEEP_DIR}/benchmark_table.csv"
