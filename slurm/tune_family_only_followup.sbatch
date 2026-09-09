#!/bin/bash
#SBATCH --job-name=dimred_tune_family_only_followup
#SBATCH --partition=GPU-a100,GPU-a40,GPU-a100s
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --array=0-1

# Follow-up round to the family_only sweep (see
# experiments/family_only_sweep_notes.md, points 2 and 4 -- not tracked by
# git). One independent GPU job per experiment, via SLURM_ARRAY_TASK_ID:
#   task 0 -- point 2: does supcon.lambda_norm > 0 fix distance: euclidean's
#             near-random collapse? (6 runs)
#   task 1 -- point 4: is the vae beta=2.0/lambda_family=0.5 dead cell a
#             systematic effect or seed sensitivity? (3 runs)
# Both experiments share the exact same pyxtal:/soap:/augmentation: fragment
# as the original configs/tuning_sweep_{vae,autoencoder,supcon}_family_only
# .example.yaml sweeps, so they hash to the identical dataset cache key --
# pointed at that already-warm shared cache below, no new warmup job needed
# (unlike the original round, which needed slurm/warmup_shared_dataset_family_only
# .sbatch to run first).
#
# Submit directly, no dependency needed:
#   sbatch slurm/tune_family_only_followup.sbatch
#
# To also run the point 3 visualization-tail experiment (independent of
# these two, since it only needs already-completed runs):
#   sbatch slurm/train_visualization_tails_family_only.sbatch

set -euo pipefail

echo "$(hostname) - array task ${SLURM_ARRAY_TASK_ID} (family_only follow-up)"

# .novalocal nodes are a known-flaky node class on this cluster (recurring
# CUDA/jaxlib and /tmp-permission failures unrelated to dim_red itself) --
# resubmit just this array task excluding this node and exit this attempt
# cleanly. Same guard as tune_family_only_models.sbatch (see that file for
# why this can't be `scontrol update ExcludeNodes` + `scontrol requeue` on
# the current job). Capped so a cluster with nothing but .novalocal nodes
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
        "${SLURM_SUBMIT_DIR}/slurm/tune_family_only_followup.sbatch"
    exit 0
fi

source ~/miniconda3/bin/activate
conda activate dmred
if [ "${CONDA_DEFAULT_ENV:-}" != "dmred" ]; then
    echo "ERROR: dmred conda environment did not activate (got '${CONDA_DEFAULT_ENV:-unset}')" >&2
    exit 1
fi

cd "${SLURM_SUBMIT_DIR}"

EXPERIMENTS=(supcon_euclidean_normfix vae_beta2_lf0.5_seedcheck)
EXPERIMENT="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"
SWEEP_CONFIG="configs/tuning_sweep_${EXPERIMENT}.example.yaml"

echo "Task ${SLURM_ARRAY_TASK_ID}: ${EXPERIMENT} (${SWEEP_CONFIG})"
dimred-sweep "${SWEEP_CONFIG}" --cache-dir runs/_shared_dataset_cache_family_only

# dimred-sweep doesn't print the sweep directory itself (only each run dir),
# so recover it as the most recently created dated dir under this
# experiment's own output_dir -- safe since this is the only job writing
# there for this experiment.
SWEEP_DIR=$(ls -td "runs/tuning_${EXPERIMENT}"/*/ | head -1)
SWEEP_DIR=${SWEEP_DIR%/}
echo "Sweep finished: ${SWEEP_DIR}"

dimred-benchmark "${SWEEP_DIR}" --output "${SWEEP_DIR}/benchmark_table.csv"
echo "Benchmark table ready: ${SWEEP_DIR}/benchmark_table.csv"
