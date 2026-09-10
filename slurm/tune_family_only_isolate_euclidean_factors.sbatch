#!/bin/bash
#SBATCH --job-name=dimred_isolate_euclidean_factors
#SBATCH --partition=GPU-a100,GPU-a40,GPU-a100s
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00

# Follow-up to the family_only sweep's point 2 and the 20260812-1 old-sweep
# comparison (see experiments/family_only_sweep_notes.md, not tracked by
# git): isolates whether train.batch_size or soap.element_agnostic (or
# both) is what actually rescues supcon.distance: euclidean's near-random
# accuracy collapse -- a 2x2 factorial, 4 runs, single job (small enough
# not to need an array).
#
# Only element_agnostic: true rows need a fresh SOAP computation (a new
# dataset cache key distinct from the shared runs/_shared_dataset_cache_family_only
# entries) -- still pointed at that same --cache-dir below so the new entry
# lands alongside the existing ones and both element_agnostic: false rows
# stay cache hits.
#
# Submit directly, no dependency needed:
#   sbatch slurm/tune_family_only_isolate_euclidean_factors.sbatch

set -euo pipefail

echo "$(hostname) - isolate euclidean factors (family_only follow-up)"

# .novalocal nodes are a known-flaky node class on this cluster (recurring
# CUDA/jaxlib and /tmp-permission failures unrelated to dim_red itself) --
# resubmit a fresh job excluding this node and exit this attempt cleanly.
# Same guard as the other slurm/*.sbatch scripts in this project. Capped so
# a cluster with nothing but .novalocal nodes free still fails loudly
# instead of looping forever.
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
        "${SLURM_SUBMIT_DIR}/slurm/tune_family_only_isolate_euclidean_factors.sbatch"
    exit 0
fi

source ~/miniconda3/bin/activate
conda activate dmred
if [ "${CONDA_DEFAULT_ENV:-}" != "dmred" ]; then
    echo "ERROR: dmred conda environment did not activate (got '${CONDA_DEFAULT_ENV:-unset}')" >&2
    exit 1
fi

cd "${SLURM_SUBMIT_DIR}"

SWEEP_CONFIG="configs/tuning_sweep_supcon_euclidean_factors.example.yaml"
dimred-sweep "${SWEEP_CONFIG}" --cache-dir runs/_shared_dataset_cache_family_only

SWEEP_DIR=$(ls -td "runs/tuning_supcon_euclidean_factors"/*/ | head -1)
SWEEP_DIR=${SWEEP_DIR%/}
echo "Sweep finished: ${SWEEP_DIR}"

# --key-hyperparams set explicitly (not the CLI default) so the resulting
# table already carries train.batch_size/soap.element_agnostic without
# needing to be regenerated afterwards.
dimred-benchmark "${SWEEP_DIR}" --output "${SWEEP_DIR}/benchmark_table.csv" \
    --key-hyperparams model,supcon.distance,supcon.tau,supcon.lambda_norm,train.batch_size,soap.element_agnostic
echo "Benchmark table ready: ${SWEEP_DIR}/benchmark_table.csv"
