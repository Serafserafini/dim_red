#!/bin/bash
#SBATCH --job-name=dimred_tune
#SBATCH --partition=GPU-a100,GPU-a40,GPU-a100s
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --array=0-0

# One FullStack sweep (dimred-sweep: family + expert stacks, body + heads
# "default" per grid combination) on one GPU with a full 2-day budget, then
# dimred-benchmark on the resulting sweep dir, so a ready benchmark_table.csv
# (see src/dim_red/pipeline/CLAUDE.md's "pipeline.benchmark" section for what's
# in it -- family classification accuracy and 2D phase-separation lead every
# row) is waiting once the job finishes. Kept as a single-task array (0-0) so
# the .novalocal resubmit guard below can resubmit "this array task".
#
# The sweep config defaults to configs/full_stack_sweep.example.yaml; point it
# at your own with --export:
#   sbatch --export=ALL,SWEEP_CONFIG=experiments/my_sweep.yaml slurm/tune_all_models.sbatch
# Datasets are built lazily per stack into the shared --cache-dir below (no
# separate warm-up job), so several sweeps over the same data reuse them.
#
# cgcnn is not part of this sweep: it is trained with dimred-run
# (configs/single_run_cgcnn.example.yaml).

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

SWEEP_CONFIG="${SWEEP_CONFIG:-configs/full_stack_sweep.example.yaml}"
echo "FullStack sweep: ${SWEEP_CONFIG}"
dimred-sweep "${SWEEP_CONFIG}" --cache-dir runs/_shared_dataset_cache

# dimred-sweep doesn't print the sweep directory itself (only each run dir),
# so recover it as the most recently created dated dir under the sweep
# config's base.output_dir -- safe as long as this is the only job writing
# there.
OUTPUT_DIR=$(python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["base"].get("output_dir", "runs"))' "${SWEEP_CONFIG}")
SWEEP_DIR=$(ls -td "${OUTPUT_DIR}"/[0-9]*-[0-9]*/ | head -1)
SWEEP_DIR=${SWEEP_DIR%/}
echo "Sweep finished: ${SWEEP_DIR}"

dimred-benchmark "${SWEEP_DIR}" --output "${SWEEP_DIR}/benchmark_table.csv"
echo "Benchmark table ready: ${SWEEP_DIR}/benchmark_table.csv"
