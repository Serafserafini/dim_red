#!/bin/bash
#SBATCH --job-name=dimred_viz_tails_family_only
#SBATCH --partition=GPU-a100,GPU-a40,GPU-a100s
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

# Follow-up round to the family_only sweep (see
# experiments/family_only_sweep_notes.md, point 3 -- not tracked by git):
# trains a real, learned 2D visualization tail (dim_red.supcon.tail_training
# .train_visualization_tail, now usable on any model_kind since
# feature/tail_for_all) on top of the best already-completed run of each of
# the three model kinds compared in this project, using the exact same
# shared tail config for all three (only the run_dir argument differs) --
# see configs/tail_train_visualization_family_only.example.yaml's own header
# comment for why cosine (not euclidean) is used here, and the exact
# accuracy each of these three runs achieved.
#
# Cheap relative to a sweep: no SOAP recompute, no body forward pass, no
# dataset cache needed at all -- dimred-train-tail only ever reads
# config.yaml/embeddings.npz from each run_dir directly. Still requests a
# GPU since the tail config's train.device is "gpu".
#
# Independent of slurm/tune_family_only_followup.sbatch (points 2/4) --
# needs only the three already-completed runs named below, submit any time:
#   sbatch slurm/train_visualization_tails_family_only.sbatch

set -euo pipefail

echo "$(hostname) - visualization tails (family_only follow-up, point 3)"

# .novalocal nodes are a known-flaky node class on this cluster (recurring
# CUDA/jaxlib and /tmp-permission failures unrelated to dim_red itself) --
# resubmit a fresh job excluding this node and exit this attempt cleanly.
# Same guard as the other slurm/*.sbatch scripts in this project. Capped so
# a cluster with nothing but .novalocal nodes free still fails loudly
# instead of looping forever.
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
        "${SLURM_SUBMIT_DIR}/slurm/train_visualization_tails_family_only.sbatch"
    exit 0
fi

source ~/miniconda3/bin/activate
conda activate dmred
if [ "${CONDA_DEFAULT_ENV:-}" != "dmred" ]; then
    echo "ERROR: dmred conda environment did not activate (got '${CONDA_DEFAULT_ENV:-unset}')" >&2
    exit 1
fi

cd "${SLURM_SUBMIT_DIR}"

TAIL_CONFIG="configs/tail_train_visualization_family_only.example.yaml"

AUTOENCODER_RUN="runs/tuning_autoencoder_family_only/20260908-1/model-autoencoder_hd-128-64_pyxtal-cub-hex-mon-ort-tet-tri-tri_nsp2_aux-family_only_lf0.5"
VAE_RUN="runs/tuning_vae_family_only/20260908-1/hd-128-64_pyxtal-cub-hex-mon-ort-tet-tri-tri_nsp2_aux-family_only_lf2"
SUPCON_RUN="runs/tuning_supcon_family_only/20260908-1/model-supcon_hd-128-64_pyxtal-cub-hex-mon-ort-tet-tri-tri_nsp2_supcon-family_only_tau0.1_lf1-3"

for RUN_DIR in "${AUTOENCODER_RUN}" "${VAE_RUN}" "${SUPCON_RUN}"; do
    echo "Training visualization tail on ${RUN_DIR}"
    dimred-train-tail "${TAIL_CONFIG}" "${RUN_DIR}"
done

echo "All three visualization tails trained."
echo "Re-run dimred-benchmark on these same three run dirs to get updated family_2d_* metrics:"
echo "  dimred-benchmark ${AUTOENCODER_RUN} ${VAE_RUN} ${SUPCON_RUN} --output runs/family_only_visualization_tail_benchmark.csv --plot --plot-data-file true"
