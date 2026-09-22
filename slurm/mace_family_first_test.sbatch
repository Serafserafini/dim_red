#!/bin/bash
#SBATCH --job-name=dimred_mace_family_first_test
#SBATCH --partition=GPU-a100,GPU-a40,GPU-a100s
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

# First cluster-scale test of model_kind: mace (frozen, pretrained MACE-MP-0
# embedding) as a drop-in replacement for the SupCon encoder+projection body,
# on the family-only classification/visualization pipeline -- see
# experiments/mace_family_first_test.yaml for the config and comparison
# target (round 15's best_combo). Runs from this worktree
# (~/code/dim_red-mace_embedding, mace_embedding branch), not the shared
# ~/code/dim_red checkout other instances/jobs use.

set -euo pipefail

echo "$(hostname) - mace family_only first test"

# .novalocal nodes are a known-flaky node class on this cluster (recurring
# CUDA/jaxlib and /tmp-permission failures unrelated to dim_red itself).
# Same guard as every other slurm/*.sbatch script in this project.
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
        "${SLURM_SUBMIT_DIR}/slurm/mace_family_first_test.sbatch"
    exit 0
fi

source ~/miniconda3/bin/activate
conda activate dmred
if [ "${CONDA_DEFAULT_ENV:-}" != "dmred" ]; then
    echo "ERROR: dmred conda environment did not activate (got '${CONDA_DEFAULT_ENV:-unset}')" >&2
    exit 1
fi

cd "${SLURM_SUBMIT_DIR}"

# This worktree's own src/ -- the editable dmred install otherwise resolves
# `import dim_red` to the main ~/code/dim_red checkout, not this worktree's
# branch (see AGENTS/session setup notes for the worktree isolation reason).
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"

python3 -c "import jax; print('jax devices:', jax.devices())"

dimred-run experiments/mace_family_first_test.yaml

echo "Run complete."
