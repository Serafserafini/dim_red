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
# we land on one, exclude it from this array task's job record and requeue
# just this task -- SLURM_JOB_ID is this specific array task's own job id
# (distinct from SLURM_ARRAY_JOB_ID, the shared parent id), so this never
# touches the other array tasks. Capped via SLURM_RESTART_COUNT (auto-set by
# SLURM, incremented on every requeue) so a cluster with nothing but
# .novalocal nodes free still fails loudly instead of looping forever.
if [[ "$(hostname)" == *.novalocal ]]; then
    MAX_RETRIES=5
    RESTART_COUNT="${SLURM_RESTART_COUNT:-0}"
    if [ "${RESTART_COUNT}" -ge "${MAX_RETRIES}" ]; then
        echo "ERROR: still landing on .novalocal nodes after ${RESTART_COUNT} retries, giving up on $(hostname)" >&2
        exit 1
    fi
    echo "$(hostname) is a .novalocal node -- excluding it and requeuing this array task (retry ${RESTART_COUNT}/${MAX_RETRIES})"
    EXISTING_EXCLUDES=$(scontrol show job -o "${SLURM_JOB_ID}" | grep -oP 'ExcludeNodes=\K\S+' || true)
    if [ -z "${EXISTING_EXCLUDES}" ] || [ "${EXISTING_EXCLUDES}" = "(null)" ]; then
        NEW_EXCLUDES="$(hostname -s)"
    else
        NEW_EXCLUDES="${EXISTING_EXCLUDES},$(hostname -s)"
    fi
    scontrol update jobid="${SLURM_JOB_ID}" ExcludeNodes="${NEW_EXCLUDES}"
    scontrol requeue "${SLURM_JOB_ID}"
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
