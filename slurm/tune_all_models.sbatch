#!/bin/bash
#SBATCH --job-name=dimred_tune
#SBATCH --partition=GPU-a100,GPU-a40,GPU-a100s
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --array=0-3

# One independent GPU job per model_kind (vae/autoencoder/supcon/cgcnn), via
# SLURM_ARRAY_TASK_ID -- each gets its own GPU and its own full 2-day
# budget, so a slow sweep for one kind can never eat into another's time.
# Each task: dimred-sweep on that kind's slurm/tuning_sweep_<kind>.yaml,
# then dimred-benchmark on the resulting sweep dir, so a ready benchmark_table.csv
# (see src/dim_red/pipeline/CLAUDE.md's "pipeline.benchmark" section for what's
# in it -- family classification accuracy and 2D phase-separation lead every
# row) is waiting for each kind once the job finishes.
#
# vae/autoencoder/supcon's tuning_sweep_*.yaml configs share an identical
# pyxtal + soap + augmentation block, so their dataset cache key is
# identical too -- they're pointed at one shared --cache-dir (below) so the
# ~35000-structure SOAP computation runs once instead of three times.
# REQUIRES that shared cache to already be built and complete before this
# array starts (np.savez isn't atomic, so concurrent array tasks racing a
# cache miss could read a half-written file) -- run warmup_shared_dataset.sbatch
# first and chain this job on it:
#   WARMUP_ID=$(sbatch --parsable slurm/warmup_shared_dataset.sbatch)
#   sbatch --dependency=afterok:${WARMUP_ID} slurm/tune_all_models.sbatch
# cgcnn (task 3) is unaffected -- it featurizes with graphs, not SOAP, so it
# keeps its own default cache dir (runs/tuning_cgcnn/_dataset_cache).
#
# To rebuild/refresh a cross-model table combining all four afterwards:
#   dimred-benchmark runs/tuning_vae/<date>-<n> runs/tuning_autoencoder/<date>-<n> \
#       runs/tuning_supcon/<date>-<n> runs/tuning_cgcnn/<date>-<n> \
#       --output runs/final_benchmark.csv

set -euo pipefail

echo "$(hostname) - array task ${SLURM_ARRAY_TASK_ID}"

# .novalocal nodes are a known-flaky node class on this cluster (recurring
# CUDA/jaxlib and /tmp-permission failures unrelated to dim_red itself). If
# we land on one, resubmit just this array task with that node excluded and
# exit this attempt cleanly.
#
# NOT `scontrol update jobid=... ExcludeNodes=...` + `scontrol requeue`: by
# the time this line runs the job is already RUNNING (that's how we know
# which node we landed on), and this cluster refuses to update a running
# job's ExcludeNodes ("Update of this parameter is not supported") -- so
# that combination always aborted (via set -e) instead of retrying. A fresh
# `sbatch --exclude=...` works because that job hasn't been scheduled yet.
# Since a fresh submission is a brand-new job with no memory of this one,
# the exclude list and retry count are threaded through via --export
# (colon-separated in DIMRED_NOVALOCAL_EXCLUDE, not comma-separated, so it
# survives being embedded inside --export's own comma-separated VAR=val
# list) -- capped so a cluster with nothing but .novalocal nodes free still
# fails loudly instead of looping forever.
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
        "${SLURM_SUBMIT_DIR}/slurm/tune_all_models.sbatch"
    exit 0
fi

source ~/miniconda3/bin/activate
conda activate dmred
if [ "${CONDA_DEFAULT_ENV:-}" != "dmred" ]; then
    echo "ERROR: dmred conda environment did not activate (got '${CONDA_DEFAULT_ENV:-unset}')" >&2
    exit 1
fi

cd "${SLURM_SUBMIT_DIR}"

MODEL_KINDS=(vae autoencoder supcon cgcnn)
MODEL_KIND="${MODEL_KINDS[$SLURM_ARRAY_TASK_ID]}"
SWEEP_CONFIG="slurm/tuning_sweep_${MODEL_KIND}.yaml"

echo "Task ${SLURM_ARRAY_TASK_ID}: tuning model_kind=${MODEL_KIND} (${SWEEP_CONFIG})"
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
