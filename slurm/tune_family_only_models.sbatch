#!/bin/bash
#SBATCH --job-name=dimred_tune_family_only
#SBATCH --partition=GPU-a100,GPU-a40,GPU-a100s
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2

# One independent GPU job per model_kind (vae/autoencoder/supcon -- no
# cgcnn, no mace, per this round's scope), via SLURM_ARRAY_TASK_ID -- each
# gets its own GPU and its own full 2-day budget, so a slow sweep for one
# kind can never eat into another's time. This is a family_only-round
# sibling of tune_all_models.sbatch: same structure, pointed at the
# configs/tuning_sweep_*_family_only.example.yaml configs (spacegroup
# ignored, all 7 pyxtal families, augmentation on) and their own shared
# dataset cache, so it never interferes with the production
# family_and_spacegroup sweeps tune_all_models.sbatch runs.
#
# Each task: dimred-sweep on that kind's own family_only config, then
# dimred-benchmark on the resulting sweep dir, so a ready
# benchmark_table.csv is waiting for each kind once the job finishes (see
# src/dim_red/pipeline/CLAUDE.md's "pipeline.benchmark" section for what's
# in it -- family classification accuracy and 2D phase-separation lead
# every row).
#
# All three configs share an identical pyxtal + soap + augmentation
# block, so their dataset cache key is identical too -- they're pointed at
# one shared --cache-dir (below) so the ~17500-structure SOAP computation
# runs once instead of three times. REQUIRES that shared cache to already
# be built and complete before this array starts (np.savez isn't atomic,
# so concurrent array tasks racing a cache miss could read a half-written
# file) -- run warmup_shared_dataset_family_only.sbatch first and chain
# this job on it:
#   WARMUP_ID=$(sbatch --parsable slurm/warmup_shared_dataset_family_only.sbatch)
#   sbatch --dependency=afterok:${WARMUP_ID} slurm/tune_family_only_models.sbatch
#
# To rebuild/refresh a cross-model table combining all three afterwards:
#   dimred-benchmark runs/tuning_vae_family_only/<date>-<n> \
#       runs/tuning_autoencoder_family_only/<date>-<n> \
#       runs/tuning_supcon_family_only/<date>-<n> \
#       --output runs/family_only_benchmark.csv --plot --plot-data-file true

set -euo pipefail

echo "$(hostname) - array task ${SLURM_ARRAY_TASK_ID} (family_only)"

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
        "${SLURM_SUBMIT_DIR}/slurm/tune_family_only_models.sbatch"
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
SWEEP_CONFIG="configs/tuning_sweep_${MODEL_KIND}_family_only.example.yaml"

echo "Task ${SLURM_ARRAY_TASK_ID}: tuning model_kind=${MODEL_KIND} (${SWEEP_CONFIG})"
dimred-sweep "${SWEEP_CONFIG}" --cache-dir runs/_shared_dataset_cache_family_only

# dimred-sweep doesn't print the sweep directory itself (only each run dir),
# so recover it as the most recently created dated dir under this kind's
# output_dir -- safe since this is the only job writing there for this kind.
SWEEP_DIR=$(ls -td "runs/tuning_${MODEL_KIND}_family_only"/*/ | head -1)
SWEEP_DIR=${SWEEP_DIR%/}
echo "Sweep finished: ${SWEEP_DIR}"

dimred-benchmark "${SWEEP_DIR}" --output "${SWEEP_DIR}/benchmark_table.csv"
echo "Benchmark table ready: ${SWEEP_DIR}/benchmark_table.csv"
