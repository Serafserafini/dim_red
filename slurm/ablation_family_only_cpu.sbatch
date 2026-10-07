#!/bin/bash
#SBATCH --job-name=dimred_ablation_family_only
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

# CPU job for the family-only ablation's dimred-run runs -- SOAP featurization is
# CPU-only regardless, and these bodies (small MLP encoder, a few thousand
# structures, 200 epochs) don't need a GPU. Pass the config path via
# CONFIG_YAML (--export=CONFIG_YAML=...), e.g.:
#
#   sbatch --job-name=dimred_ablation_aug_off \
#       --export=ALL,CONFIG_YAML=experiments/ablation_family_only/aug_off.yaml \
#       slurm/ablation_family_only_cpu.sbatch
#
# Same .novalocal retry guard as tune_all_models.sbatch (see repo-root
# CLAUDE.md / session notes: jobs landing on .novalocal hostnames recur with
# infra errors, e.g. /tmp PTX write PERMISSION_DENIED -- suspect the node
# before dim_red code).

set -euo pipefail

: "${CONFIG_YAML:?CONFIG_YAML must be set via --export}"

echo "$(hostname) - ablation_family_only: ${CONFIG_YAML}"

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
        --job-name="${SLURM_JOB_NAME}" \
        --exclude="${EXCLUDE_LIST//:/,}" \
        --export="ALL,CONFIG_YAML=${CONFIG_YAML},DIMRED_NOVALOCAL_EXCLUDE=${EXCLUDE_LIST},DIMRED_NOVALOCAL_RETRY=$((RETRY_COUNT + 1))" \
        "${SLURM_SUBMIT_DIR}/slurm/ablation_family_only_cpu.sbatch"
    exit 0
fi

source ~/miniconda3/bin/activate
conda activate dmred
if [ "${CONDA_DEFAULT_ENV:-}" != "dmred" ]; then
    echo "ERROR: dmred conda environment did not activate (got '${CONDA_DEFAULT_ENV:-unset}')" >&2
    exit 1
fi

cd "${SLURM_SUBMIT_DIR}"

# Run from a git worktree whose editable `pip install -e .` still resolves
# to a DIFFERENT checkout's src/ (an editable install records an absolute
# path, independent of cwd) -- prepend this worktree's own src/ so
# `dimred-run` actually executes *this* checkout's code, not whatever
# happened to be installed when `pip install -e .` was last run elsewhere.
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

dimred-run "${CONFIG_YAML}"

echo "Run complete."
