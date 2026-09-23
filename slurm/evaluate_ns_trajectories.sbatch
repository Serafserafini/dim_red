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
# ns.<i>.traj.extxyz) with the round 15 best_combo SupCon body +
# hierarchical_supcon tail + round 16 SG visualizers -- see
# examples/evaluate_ns_trajectories.py for everything that gets saved.
# ~2-3 min per replica (44k frames of 32 atoms) on one core; SOAP + tiny
# MLPs only, so CPU is enough (no GPU, and JAX_PLATFORMS=cpu below).
#
# No .novalocal resubmit guard here, unlike the GPU jobs: that failure mode
# (/tmp PTX writes) is GPU-only.
#
# Expects the trajectories at experiments/test_strucutres/ti_trajs (rsync'd,
# gitignored) and writes runs/ns_traj_eval/ti_trajs/ns.<i>.eval.npz.
# Resumable: re-submitting skips finished replicas and finished chunks.
#
# Submit from repo root:
#   sbatch slurm/evaluate_ns_trajectories.sbatch

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

python examples/evaluate_ns_trajectories.py \
    --replica "${SLURM_ARRAY_TASK_ID}" \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}"

echo "Replica ${SLURM_ARRAY_TASK_ID} done: ${OUTPUT_DIR}/ns.${SLURM_ARRAY_TASK_ID}.eval.npz"
