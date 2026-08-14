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
# Submit from the repo root:
#   sbatch slurm/tune_all_models.sbatch
#
# To rebuild/refresh a cross-model table combining all four afterwards:
#   dimred-benchmark runs/tuning_vae/<date>-<n> runs/tuning_autoencoder/<date>-<n> \
#       runs/tuning_supcon/<date>-<n> runs/tuning_cgcnn/<date>-<n> \
#       --output runs/final_benchmark.csv

set -euo pipefail

echo "$(hostname) - array task ${SLURM_ARRAY_TASK_ID}"

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
dimred-sweep "${SWEEP_CONFIG}"

# dimred-sweep doesn't print the sweep directory itself (only each run dir),
# so recover it as the most recently created dated dir under this kind's
# output_dir -- safe since this is the only job writing there for this kind.
SWEEP_DIR=$(ls -td "runs/tuning_${MODEL_KIND}"/*/ | head -1)
SWEEP_DIR=${SWEEP_DIR%/}
echo "Sweep finished: ${SWEEP_DIR}"

dimred-benchmark "${SWEEP_DIR}" --output "${SWEEP_DIR}/benchmark_table.csv"
echo "Benchmark table ready: ${SWEEP_DIR}/benchmark_table.csv"
