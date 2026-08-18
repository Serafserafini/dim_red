#!/bin/bash
#SBATCH --job-name=dimred_tune_single
#SBATCH --partition=GPU-a100,GPU-a40,GPU-a100s
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00

set -euo pipefail

echo "$(hostname)"


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
    echo "$(hostname) is a .novalocal node -- resubmitting task excluding [${EXCLUDE_LIST}] (retry $((RETRY_COUNT + 1))/${MAX_RETRIES})"
    sbatch \
        --exclude="${EXCLUDE_LIST//:/,}" \
        --export="ALL,DIMRED_NOVALOCAL_EXCLUDE=${EXCLUDE_LIST},DIMRED_NOVALOCAL_RETRY=$((RETRY_COUNT + 1))" \
        "${SLURM_SUBMIT_DIR}/slurm/tune_single_model.sbatch"
    exit 0
fi

source ~/miniconda3/bin/activate
conda activate dmred
if [ "${CONDA_DEFAULT_ENV:-}" != "dmred" ]; then
    echo "ERROR: dmred conda environment did not activate (got '${CONDA_DEFAULT_ENV:-unset}')" >&2
    exit 1
fi

cd "${SLURM_SUBMIT_DIR}"

# take the model kind from command line argument (e.g. `sbatch --export=MODEL_KIND=vae slurm/tune_single_model.sbatch`)
MODEL_KIND="${MODEL_KIND:-}"
SWEEP_CONFIG="slurm/tuning_sweep_${MODEL_KIND}.yaml"

echo "Tuning model_kind=${MODEL_KIND} (${SWEEP_CONFIG})"
if [ "${MODEL_KIND}" = "cgcnn" ]; then
    # No shared cache for cgcnn: it builds graph features, not SOAP, so
    # there's nothing to share with the other three kinds. Default
    # (runs/tuning_cgcnn/_dataset_cache) is fine.
    dimred-sweep "${SWEEP_CONFIG}"
else
    # vae/autoencoder/supcon share one dataset cache -- see the file-level
    # comment above. Must already be warmed up by warmup_shared_dataset.sbatch.
    dimred-sweep "${SWEEP_CONFIG}" --cache-dir runs/_shared_dataset_cache
fi

# dimred-sweep doesn't print the sweep directory itself (only each run dir),
# so recover it as the most recently created dated dir under this kind's
# output_dir -- safe since this is the only job writing there for this kind.
SWEEP_DIR=$(ls -td "runs/tuning_${MODEL_KIND}"/*/ | head -1)
SWEEP_DIR=${SWEEP_DIR%/}
echo "Sweep finished: ${SWEEP_DIR}"

dimred-benchmark "${SWEEP_DIR}" --output "${SWEEP_DIR}/benchmark_table.csv"
echo "Benchmark table ready: ${SWEEP_DIR}/benchmark_table.csv"
