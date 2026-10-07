#!/bin/bash
#SBATCH --job-name=dimred_eval_ns_trajs
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-15
#SBATCH --output=slurm-%x-%A_%a.out

# Evaluates EVERY frame of one NS replica trajectory (array task i ->
# ns.<i>.traj.extxyz) with a trained FullStack run (family stack + every
# trained expert stack, one heads set) -- see
# examples/evaluate_ns_trajectories.py for everything that gets saved.
# SOAP + tiny MLPs only, so CPU is enough (no GPU, and JAX_PLATFORMS=cpu
# below).
#
# The run is REQUIRED (DIMRED_RUN_DIR); DIMRED_HEADS_NAME is optional
# (default: the only heads set each stack has):
#   sbatch --export=ALL,DIMRED_RUN_DIR=runs/<full_stack_run> \
#       slurm/evaluate_ns_trajectories.sbatch
#
# No .novalocal resubmit guard here, unlike the GPU jobs: that failure mode
# (/tmp PTX writes) is GPU-only.
#
# Expects the trajectories at experiments/test_strucutres/ti_trajs (rsync'd,
# gitignored) and writes runs/ns_traj_eval/ti_trajs/ns.<i>.eval.npz.
# Resumable: re-submitting skips finished replicas and finished chunks.
#
# Submit from repo root (see the --export example above).

set -euo pipefail

echo "$(hostname) - array task ${SLURM_ARRAY_TASK_ID} (NS trajectory evaluation)"

source ~/miniconda3/bin/activate
conda activate dmred
if [ "${CONDA_DEFAULT_ENV:-}" != "dmred" ]; then
    echo "ERROR: dmred conda environment did not activate (got '${CONDA_DEFAULT_ENV:-unset}')" >&2
    exit 1
fi

cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

INPUT_DIR="${DIMRED_NS_INPUT_DIR:-experiments/test_strucutres/ti_trajs}"
OUTPUT_DIR="${DIMRED_NS_OUTPUT_DIR:-runs/ns_traj_eval/ti_trajs}"
: "${DIMRED_RUN_DIR:?DIMRED_RUN_DIR (a FullStack run directory) must be set via --export}"
HEADS_ARGS=()
if [ -n "${DIMRED_HEADS_NAME:-}" ]; then
    HEADS_ARGS=(--heads-name "${DIMRED_HEADS_NAME}")
fi

python examples/evaluate_ns_trajectories.py \
    --replica "${SLURM_ARRAY_TASK_ID}" \
    --run-dir "${DIMRED_RUN_DIR}" \
    "${HEADS_ARGS[@]}" \
    --device cpu \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}"

echo "Replica ${SLURM_ARRAY_TASK_ID} done: ${OUTPUT_DIR}/ns.${SLURM_ARRAY_TASK_ID}.eval.npz"
