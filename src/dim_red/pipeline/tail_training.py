"""
Phase 2 pipeline entry point: freeze an already-trained run's body and train
exactly one tail (classification, visualization, or hierarchical) on its
saved representations -- see ``dim_red.supcon.tail_training`` for the actual
training loops (reused as-is regardless of which body produced the
representations, since neither training loop ever touches the body itself),
and ``dim_red.pipeline.inference`` for ``load_run_embeddings``, the
lightweight loader used here: this module only ever needs a run's
``config.yaml``/``embeddings.npz``, never the reconstructed body itself (no
forward pass, no encoding), so it deliberately does *not* use
``load_trained_run`` (which would also reconstruct the model, resolve
species, and compute/recompute standardization stats -- all wasted work for
training a tail on already-saved representations).

Which ``model_kind``s a given ``tail_kind`` accepts is gated per-tail-kind by
``_TAIL_MODEL_KINDS`` below, not by one blanket allowlist: ``"visualization"``
works for *any* model_kind, including ``"vae"``/``"autoencoder"`` -- their
``aux_heads`` classification heads have no equivalent low-dimensional
*visualization* projection of their own, so a visualization tail there is a
genuinely new capability, not a duplicate of anything they already produce.
``"classification"``/``"hierarchical"`` still require ``"supcon"``/
``"cgcnn"``/``"mace"``: a vae/autoencoder body already has its own
classification heads via ``aux_heads``, so those two tail kinds would be
pure duplication for it (same reasoning ``cgcnn``'s own classification tail
support already documents -- redundant with its built-in heads, but harmless
for ablation/comparison; that reasoning simply doesn't extend to
vae/autoencoder, which have no separate tail-training workflow for anything
*except* visualization).

Reuses phase 1's exact train/val split (``embeddings.npz["split"]``) and
representations (``embeddings.npz["embeddings"]``) -- no SOAP recompute ever
(not even the ``load_trained_run`` fallback for pre-existing runs, since
this module never calls it), no body forward pass at all, so this is freely
rerunnable with different tail configs against the same trained body --
including runs whose ``dataset.extxyz``/``model_params.msgpack`` are no
longer present, since neither is required here.

``tail_kind == "hierarchical"`` (see ``_train_hierarchical_tail``) is a
genuinely two-stage classifier, as opposed to ``"classification"``'s single
family-masked spacegroup head (``dim_red.supcon.tails.apply_family_mask``):
one ``ClassificationTail`` predicts family, then one independent,
separately-trained ``ClassificationTail`` *per family* ("expert") predicts
spacegroup, trained only on that family's own rows and only over the
spacegroups actually observed within it. Reuses ``ClassificationTail``/
``train_classification_tail`` unchanged -- both are already generic
single-head classifiers when built without a spacegroup head at all; a
hierarchical expert simply reuses that same "family" head slot to mean
"spacegroup, local to this one family".
"""

from __future__ import annotations

import csv
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from flax import serialization

from dim_red.analysis.plotting import (
    plot_classification_report,
    plot_confusion_matrix,
    plot_reduced_space,
    plot_reduced_space_3d,
    plot_reliability_diagram,
)
from dim_red.pipeline.config import TailTrainConfig, tail_train_config_to_dict
from dim_red.pipeline.inference import load_run_embeddings
from dim_red.supcon.tail_training import TailTrainConfig as SupConTailTrainConfig
from dim_red.supcon.tail_training import (
    train_classification_tail,
    train_visualization_tail,
)
from dim_red.supcon.tails import (
    ClassificationTail,
    VisualizationTail,
    apply_family_mask,
)

logger = logging.getLogger("dim_red.pipeline")

# Which model_kinds a given tail_kind may be trained against -- see the
# module docstring above for the reasoning. Keyed by config.TailTrainConfig
# .tail_kind's own three valid values.
_TAIL_MODEL_KINDS: Dict[str, Tuple[str, ...]] = {
    "classification": ("supcon", "cgcnn", "mace"),
    "visualization": ("supcon", "cgcnn", "mace", "vae", "autoencoder"),
    "hierarchical": ("supcon", "cgcnn", "mace"),
}


def _make_unique_run_dir(output_dir: Path, name: str) -> Path:
    """Create and return ``output_dir / name``, deduplicating with a
    ``-<n>`` suffix if that directory already exists -- duplicated from
    ``dim_red.pipeline.single_run`` (small private helper, same precedent
    as ``_iter_batches`` being duplicated elsewhere in this codebase).
    """
    run_dir = output_dir / name
    suffix = 1
    while True:
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            suffix += 1
            run_dir = output_dir / f"{name}-{suffix}"


def _build_vocab_ids(values: List) -> Tuple[List, np.ndarray]:
    """Map arbitrary hashable values to a sorted vocabulary and integer ids.
    Duplicated from ``dim_red.pipeline.single_run``.

    Returns:
        ``(vocab, ids)`` where ``vocab[i]`` is the value for class id ``i``.
    """
    vocab = sorted(set(values))
    value_to_id = {v: i for i, v in enumerate(vocab)}
    ids = np.array([value_to_id[v] for v in values], dtype=np.int64)
    return vocab, ids


def _build_family_spacegroup_mask(
    family_ids: np.ndarray, spacegroup_ids: np.ndarray, n_family: int, n_spacegroup: int
) -> np.ndarray:
    """Empirical co-occurrence mask: 1.0 where a spacegroup was observed
    under a family, 0.0 otherwise. Duplicated from ``dim_red.pipeline.single_run``.
    """
    mask = np.zeros((n_family, n_spacegroup), dtype=np.float32)
    mask[family_ids, spacegroup_ids] = 1.0
    return mask


def _save_loss_history(path: Path, history: Dict[str, List[float]]) -> None:
    """Duplicated from ``dim_red.pipeline.single_run``."""
    fieldnames = ["epoch"] + list(history.keys())
    n_epochs = len(next(iter(history.values())))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(n_epochs):
            row = {"epoch": epoch + 1}
            row.update({k: v[epoch] for k, v in history.items()})
            writer.writerow(row)


def _sanitize_family_dirname(family: str) -> str:
    """Turn a family label into a filesystem-safe directory name (used
    under ``tails/hierarchical/experts/``). Family labels are always short
    human-readable strings in practice (crystal system names, e.g.
    ``"Cubic"``, or a pyxtal lattice-type string), but this sanitizes
    defensively rather than assume that always holds.
    """
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(family))
    return safe or "unknown"


def _classifier_eval_plots(
    tail_dir: Path,
    eval_targets: List[Tuple[str, np.ndarray, np.ndarray, List]],
    split_masks: Dict[str, np.ndarray],
) -> None:
    """A confusion matrix, a per-class precision/recall/F1 bar chart, and a
    calibration/reliability diagram, on every named split, for every
    ``(level, y_true_all, y_probs_all, class_names)`` entry in
    ``eval_targets`` -- shared by the ``"classification"`` and
    ``"hierarchical"`` tail kinds, both of which end up with the same
    ``labels``/``family_probs``/``spacegroups``/``spacegroup_probs``-shaped
    arrays to evaluate.
    """
    for level, y_true_all, y_probs_all, class_names in eval_targets:
        for split_name, mask in split_masks.items():
            stem = f"{level}_{split_name}"
            y_true_split = y_true_all[mask]
            y_probs_split = y_probs_all[mask]
            plot_confusion_matrix(
                y_true_split,
                y_probs_split,
                class_names,
                title=f"{level.capitalize()} confusion matrix ({split_name})",
                save_path=str(tail_dir / f"confusion_matrix_{stem}.png"),
            )
            plot_classification_report(
                y_true_split,
                y_probs_split,
                class_names,
                title=f"{level.capitalize()} precision/recall/F1 ({split_name})",
                save_path=str(tail_dir / f"classification_report_{stem}.png"),
            )
            plot_reliability_diagram(
                y_true_split,
                y_probs_split,
                class_names,
                title=f"{level.capitalize()} calibration ({split_name})",
                save_path=str(tail_dir / f"calibration_{stem}.png"),
            )
        logger.info(
            "Saved %s classifier-evaluation plots (confusion matrix, "
            "classification report, calibration -- train+val) to %s",
            level,
            tail_dir,
        )


def _train_hierarchical_tail(
    config: TailTrainConfig,
    tail_dir: Path,
    r_all: np.ndarray,
    labels_all: np.ndarray,
    spacegroups_all: np.ndarray,
    material_ids_all: np.ndarray,
    split_all: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> None:
    """Train a genuinely two-stage classifier: one ``ClassificationTail``
    (family-only mode) predicting family, then one independent
    ``ClassificationTail`` (also family-only mode -- its "family" slot is
    reused to mean "spacegroup, local to this one family") *per family*,
    each trained only on that family's own rows and only over the
    spacegroups actually observed within it -- as opposed to
    ``"classification"``'s single family-masked spacegroup head (see
    ``dim_red.supcon.tails.apply_family_mask``).

    Predictions are assembled into the exact same ``tail_predictions.npz``
    schema ``"classification"`` already produces (``family_probs``/
    ``family_classes``/``labels``, ``spacegroup_probs``/``spacegroup_classes``/
    ``spacegroups``), by scattering each expert's local softmax into the
    matching columns of a global (union) spacegroup vocabulary -- so every
    existing downstream reader (``dim_red.pipeline.compare.
    classification_accuracies_from_npz``, ``dim_red.pipeline.benchmark.
    run_classification_accuracies``, the confusion-matrix/classification-
    report/calibration plots below) works unmodified. ``spacegroup_probs``
    uses each row's *predicted* family to pick the expert (the honest,
    deployable end-to-end pipeline); the extra ``spacegroup_probs_oracle``
    key (ignored by that existing code) instead uses the row's *true*
    family, isolating expert quality from stage-1 routing quality -- see
    ``dim_red.pipeline.compare.hierarchical_accuracies_from_npz``.

    Families with fewer than ``config.hierarchical.min_samples_per_expert``
    training rows, or fewer than 2 distinct spacegroups observed within
    them, get no dedicated expert at all -- they fall back to always
    predicting that family's single most frequent training-set spacegroup
    (probability 1.0). Every family's outcome (trained expert vs. fallback,
    training-row count, local vocabulary size) is recorded in
    ``family_expert_status.yaml``.
    """
    hierarchical = config.hierarchical
    input_dim = r_all.shape[1]

    family_classes, family_ids = _build_vocab_ids(labels_all.tolist())
    logger.info("%d family classes: %s", len(family_classes), family_classes)
    spacegroup_classes, _ = _build_vocab_ids(spacegroups_all.tolist())
    spacegroup_class_to_col = {v: i for i, v in enumerate(spacegroup_classes)}
    n_spacegroup = len(spacegroup_classes)
    logger.info("%d spacegroup classes observed", n_spacegroup)

    logger.info(
        "Training hierarchical tail: head_hidden_dim=%d min_samples_per_expert=%d "
        "epochs=%d batch_size=%d device=%s optimizer=%s early_stopping=%s",
        hierarchical.head_hidden_dim,
        hierarchical.min_samples_per_expert,
        config.train.epochs,
        config.train.batch_size,
        config.train.device,
        config.train.optimizer,
        config.train.early_stopping.enabled,
    )
    train_config = SupConTailTrainConfig(
        epochs=config.train.epochs,
        batch_size=config.train.batch_size,
        learning_rate=config.train.learning_rate,
        optimizer=config.train.optimizer,
        seed=config.seed,
        device=config.train.device,
        early_stopping=config.train.early_stopping.enabled,
        early_stopping_patience=config.train.early_stopping.patience,
        early_stopping_min_delta=config.train.early_stopping.min_delta,
        early_stopping_restore_best=config.train.early_stopping.restore_best_weights,
    )

    # --- Stage 1: family ----------------------------------------------------
    family_dir = tail_dir / "family"
    family_dir.mkdir(parents=True, exist_ok=True)
    family_tail = ClassificationTail(
        input_dim=input_dim,
        hidden_dim=hierarchical.head_hidden_dim,
        n_family_classes=len(family_classes),
        seed=config.seed,
    )
    family_history = train_classification_tail(
        family_tail,
        r_all[train_idx],
        r_all[val_idx],
        train_config,
        train_family_ids=family_ids[train_idx],
        val_family_ids=family_ids[val_idx],
    )
    logger.info(
        "Family stage training complete: %d/%d epochs, final train_loss=%.4f "
        "val_loss=%.4f",
        len(family_history["train_loss"]),
        config.train.epochs,
        family_history["train_loss"][-1],
        family_history["val_loss"][-1],
    )
    _save_loss_history(family_dir / "loss_history.csv", family_history)
    with open(family_dir / "tail_params.msgpack", "wb") as f:
        f.write(serialization.to_bytes(family_tail.params))

    family_probs_all = np.asarray(
        jax.nn.softmax(family_tail.classify_family(r_all), axis=-1)
    )
    family_pred_ids = family_probs_all.argmax(axis=1)

    # --- Stage 2: one independent expert per family --------------------------
    spacegroup_probs_all = np.zeros((r_all.shape[0], n_spacegroup), dtype=np.float32)
    spacegroup_probs_oracle_all = np.zeros(
        (r_all.shape[0], n_spacegroup), dtype=np.float32
    )
    expert_status: List[Dict[str, Any]] = []

    for k, family_name in enumerate(family_classes):
        family_mask_all = family_ids == k
        family_mask_train = train_idx & family_mask_all
        family_mask_val = val_idx & family_mask_all
        local_spacegroups = spacegroups_all[family_mask_all]
        local_classes, local_ids_subset = _build_vocab_ids(local_spacegroups.tolist())
        local_ids_full = np.full(r_all.shape[0], -1, dtype=np.int64)
        local_ids_full[family_mask_all] = local_ids_subset

        n_train_k = int(family_mask_train.sum())
        predicted_mask = family_pred_ids == k
        true_mask = family_mask_all

        if n_train_k < hierarchical.min_samples_per_expert or len(local_classes) < 2:
            # Not enough data (or not enough spacegroup variety) for a
            # dedicated expert -- fall back to this family's single most
            # frequent training-set spacegroup instead.
            source_spacegroups = spacegroups_all[family_mask_train]
            if source_spacegroups.size == 0:
                source_spacegroups = spacegroups_all[family_mask_val]
            if source_spacegroups.size == 0:
                source_spacegroups = local_spacegroups
            majority_value = int(
                Counter(source_spacegroups.tolist()).most_common(1)[0][0]
            )
            expert_status.append(
                {
                    "family": family_name,
                    "expert": False,
                    "n_train": n_train_k,
                    "n_local_spacegroup_classes": len(local_classes),
                    "fallback_spacegroup": majority_value,
                }
            )
            col = spacegroup_class_to_col[majority_value]
            spacegroup_probs_all[np.ix_(predicted_mask, [col])] = 1.0
            spacegroup_probs_oracle_all[np.ix_(true_mask, [col])] = 1.0
            logger.info(
                "Family %r: %d training rows, %d local spacegroup classes -- "
                "below min_samples_per_expert=%d or <2 classes, falling back "
                "to majority spacegroup %d",
                family_name,
                n_train_k,
                len(local_classes),
                hierarchical.min_samples_per_expert,
                majority_value,
            )
            continue

        # A family with no validation rows of its own reuses its training
        # rows for "validation" too, rather than crashing on an empty split
        # -- val_loss then just tracks train_loss for that one expert.
        expert_val_mask = (
            family_mask_val if family_mask_val.any() else family_mask_train
        )
        expert_dir = tail_dir / "experts" / _sanitize_family_dirname(family_name)
        expert_dir.mkdir(parents=True, exist_ok=True)
        expert_tail = ClassificationTail(
            input_dim=input_dim,
            hidden_dim=hierarchical.head_hidden_dim,
            n_family_classes=len(local_classes),
            seed=config.seed,
        )
        expert_history = train_classification_tail(
            expert_tail,
            r_all[family_mask_train],
            r_all[expert_val_mask],
            train_config,
            train_family_ids=local_ids_full[family_mask_train],
            val_family_ids=local_ids_full[expert_val_mask],
        )
        _save_loss_history(expert_dir / "loss_history.csv", expert_history)
        with open(expert_dir / "tail_params.msgpack", "wb") as f:
            f.write(serialization.to_bytes(expert_tail.params))
        with open(expert_dir / "local_spacegroup_classes.yaml", "w") as f:
            yaml.safe_dump(
                {"local_spacegroup_classes": [int(c) for c in local_classes]}, f
            )

        expert_status.append(
            {
                "family": family_name,
                "expert": True,
                "n_train": n_train_k,
                "n_local_spacegroup_classes": len(local_classes),
            }
        )
        logger.info(
            "Family %r: trained expert on %d training rows, %d local "
            "spacegroup classes, %d/%d epochs, final train_loss=%.4f "
            "val_loss=%.4f",
            family_name,
            n_train_k,
            len(local_classes),
            len(expert_history["train_loss"]),
            config.train.epochs,
            expert_history["train_loss"][-1],
            expert_history["val_loss"][-1],
        )

        cols = [spacegroup_class_to_col[c] for c in local_classes]
        if predicted_mask.any():
            local_probs = np.asarray(
                jax.nn.softmax(
                    expert_tail.classify_family(r_all[predicted_mask]), axis=-1
                )
            )
            spacegroup_probs_all[np.ix_(predicted_mask, cols)] = local_probs
        if true_mask.any():
            local_probs_oracle = np.asarray(
                jax.nn.softmax(expert_tail.classify_family(r_all[true_mask]), axis=-1)
            )
            spacegroup_probs_oracle_all[np.ix_(true_mask, cols)] = local_probs_oracle

    with open(tail_dir / "family_expert_status.yaml", "w") as f:
        yaml.safe_dump({"families": expert_status}, f, sort_keys=False)
    logger.info(
        "Saved per-family expert/fallback status to %s",
        tail_dir / "family_expert_status.yaml",
    )

    predictions_payload = dict(
        material_ids=material_ids_all,
        split=split_all,
        labels=labels_all,
        family_probs=family_probs_all,
        family_classes=np.array(family_classes),
        spacegroups=spacegroups_all,
        spacegroup_probs=spacegroup_probs_all,
        spacegroup_probs_oracle=spacegroup_probs_oracle_all,
        spacegroup_classes=np.array(spacegroup_classes),
    )
    np.savez(tail_dir / "tail_predictions.npz", **predictions_payload)
    logger.info(
        "Saved hierarchical classification predictions to %s",
        tail_dir / "tail_predictions.npz",
    )

    eval_targets = [
        ("family", labels_all, family_probs_all, family_classes),
        ("spacegroup", spacegroups_all, spacegroup_probs_all, spacegroup_classes),
    ]
    _classifier_eval_plots(tail_dir, eval_targets, {"train": train_idx, "val": val_idx})


def train_tail(config: TailTrainConfig) -> Path:
    """Freeze a completed run's body and train exactly one tail
    (classification, visualization, or hierarchical) on its already-saved
    representations, saving the tail's own artifacts to
    ``<run_dir>/tails/<output_subdir or tail_kind>/``. Which ``model_kind``s
    are accepted depends on ``config.tail_kind`` -- see
    ``_TAIL_MODEL_KINDS``: ``"visualization"`` accepts any model_kind;
    ``"classification"``/``"hierarchical"`` still require
    ``"supcon"``/``"cgcnn"``/``"mace"``.

    Args:
        config: Which run/tail/hyperparameters to train -- see
            ``dim_red.pipeline.config.TailTrainConfig``.

    Returns:
        The tail's output directory (``run.log`` -- every log line emitted
        from here on, mirroring ``dim_red.pipeline.single_run.run_single``'s
        own ``run.log`` -- ``tail_config.yaml``, plus, for
        ``tail_kind="classification"``: ``tail_params.msgpack``,
        ``loss_history.csv``, ``tail_predictions.npz``
        (predicted probabilities, class vocabularies, and the true labels,
        for a classification tail) and a confusion matrix/precision-recall-F1
        bar chart/calibration diagram per active label level and per
        train/val split (``confusion_matrix_<level>_<split>.png``,
        ``classification_report_<level>_<split>.png``,
        ``calibration_<level>_<split>.png``); for ``tail_kind="visualization"``:
        ``tail_embeddings.npz`` plus plot(s); or for
        ``tail_kind="hierarchical"``: ``family/`` (stage-1 tail),
        ``experts/<family>/`` (one per trained stage-2 expert),
        ``family_expert_status.yaml``, ``tail_predictions.npz``, and the same
        plot suite as ``"classification"``.

    Raises:
        ValueError: If ``config.run_dir`` is unset, or wasn't produced by a
            run whose ``model_kind`` is allowed for ``config.tail_kind`` --
            see ``_TAIL_MODEL_KINDS``.
    """
    if config.run_dir is None:
        raise ValueError(
            "config.run_dir is unset -- pass a run directory on the command "
            "line (dimred-train-tail <config> <run_dir>) or set run_dir "
            "directly in the TailTrainConfig/YAML"
        )
    run_dir = Path(config.run_dir)
    loaded = load_run_embeddings(run_dir)
    allowed_model_kinds = _TAIL_MODEL_KINDS[config.tail_kind]
    if loaded.config.model_kind not in allowed_model_kinds:
        raise ValueError(
            f"{run_dir} is a model_kind={loaded.config.model_kind!r} run -- "
            f"tail_kind={config.tail_kind!r} requires a completed run whose "
            f"model_kind is one of {allowed_model_kinds} (classification/"
            "hierarchical tails are redundant with vae/autoencoder's own "
            "built-in aux_heads classification, so those two tail kinds "
            "aren't offered for them; a visualization tail has no such "
            "built-in equivalent and works for any model_kind)"
        )

    output_subdir = config.output_subdir or config.tail_kind
    tail_dir = _make_unique_run_dir(run_dir / "tails", output_subdir)

    file_handler = logging.FileHandler(tail_dir / "run.log")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logger.addHandler(file_handler)
    try:
        logger.info("Training tail_kind=%s in %s", config.tail_kind, tail_dir)

        r_all = loaded.embeddings["embeddings"]
        labels_all = loaded.embeddings["labels"]
        material_ids_all = loaded.embeddings["material_ids"]
        spacegroups_all = loaded.embeddings["spacegroups"]
        split_all = loaded.embeddings["split"]
        train_idx = split_all == "train"
        val_idx = split_all == "val"
        logger.info(
            "Loaded %d representations from %s (%d train / %d val)",
            r_all.shape[0],
            run_dir,
            int(train_idx.sum()),
            int(val_idx.sum()),
        )

        if config.tail_kind == "hierarchical":
            with open(tail_dir / "tail_config.yaml", "w") as f:
                yaml.safe_dump(tail_train_config_to_dict(config), f, sort_keys=False)
            _train_hierarchical_tail(
                config,
                tail_dir,
                r_all,
                labels_all,
                spacegroups_all,
                material_ids_all,
                split_all,
                train_idx,
                val_idx,
            )
            logger.info("Tail artifacts saved to %s", tail_dir)
        else:
            if config.tail_kind == "classification":
                mode = config.classification.mode
                use_family = True
                use_spacegroup = mode == "family_and_spacegroup"
                balanced_batching = False
            else:
                mode = config.visualization.mode
                use_family = mode != "spacegroup_only"
                use_spacegroup = mode != "family_only"
                balanced_batching = config.visualization.batching.strategy == "balanced"

            need_family_ids = use_family or balanced_batching
            need_spacegroup_ids = use_spacegroup or balanced_batching

            family_classes: List[str] = []
            spacegroup_classes: List[int] = []
            family_ids = None
            spacegroup_ids = None
            family_spacegroup_mask = None
            if need_family_ids:
                family_classes, family_ids = _build_vocab_ids(labels_all.tolist())
                logger.info(
                    "%d family classes: %s", len(family_classes), family_classes
                )
            if need_spacegroup_ids:
                spacegroup_classes, spacegroup_ids = _build_vocab_ids(
                    spacegroups_all.tolist()
                )
                logger.info("%d spacegroup classes observed", len(spacegroup_classes))
                if config.tail_kind == "classification" and use_spacegroup:
                    family_spacegroup_mask = _build_family_spacegroup_mask(
                        family_ids,
                        spacegroup_ids,
                        len(family_classes),
                        len(spacegroup_classes),
                    )

            with open(tail_dir / "tail_config.yaml", "w") as f:
                yaml.safe_dump(tail_train_config_to_dict(config), f, sort_keys=False)

            train_config = SupConTailTrainConfig(
                epochs=config.train.epochs,
                batch_size=config.train.batch_size,
                learning_rate=config.train.learning_rate,
                optimizer=config.train.optimizer,
                tau=(
                    config.visualization.tau
                    if config.tail_kind == "visualization"
                    else 0.1
                ),
                distance=(
                    config.visualization.distance
                    if config.tail_kind == "visualization"
                    else "euclidean"
                ),
                seed=config.seed,
                device=config.train.device,
                early_stopping=config.train.early_stopping.enabled,
                early_stopping_patience=config.train.early_stopping.patience,
                early_stopping_min_delta=config.train.early_stopping.min_delta,
                early_stopping_restore_best=config.train.early_stopping.restore_best_weights,
            )

            if config.tail_kind == "classification":
                tail = ClassificationTail(
                    input_dim=r_all.shape[1],
                    hidden_dim=config.classification.head_hidden_dim,
                    n_family_classes=len(family_classes) if use_family else None,
                    n_spacegroup_classes=(
                        len(spacegroup_classes) if use_spacegroup else None
                    ),
                    seed=config.seed,
                )
                logger.info(
                    "Training classification tail: mode=%s head_hidden_dim=%d epochs=%d "
                    "batch_size=%d device=%s optimizer=%s early_stopping=%s",
                    mode,
                    config.classification.head_hidden_dim,
                    config.train.epochs,
                    config.train.batch_size,
                    config.train.device,
                    config.train.optimizer,
                    config.train.early_stopping.enabled,
                )
                history = train_classification_tail(
                    tail,
                    r_all[train_idx],
                    r_all[val_idx],
                    train_config,
                    train_family_ids=family_ids[train_idx] if use_family else None,
                    val_family_ids=family_ids[val_idx] if use_family else None,
                    train_spacegroup_ids=(
                        spacegroup_ids[train_idx] if use_spacegroup else None
                    ),
                    val_spacegroup_ids=(
                        spacegroup_ids[val_idx] if use_spacegroup else None
                    ),
                    lambda_family=config.classification.lambda_family,
                    lambda_spacegroup=config.classification.lambda_spacegroup,
                    family_spacegroup_mask=family_spacegroup_mask,
                )
            else:
                tail = VisualizationTail(
                    input_dim=r_all.shape[1],
                    hidden_dim=config.visualization.hidden_dim or [r_all.shape[1]],
                    output_dim=config.visualization.viz_dim,
                    seed=config.seed,
                )
                logger.info(
                    "Training visualization tail: viz_dim=%d mode=%s tau=%.3f distance=%s "
                    "epochs=%d batch_size=%d device=%s optimizer=%s batching=%s "
                    "early_stopping=%s",
                    config.visualization.viz_dim,
                    mode,
                    config.visualization.tau,
                    config.visualization.distance,
                    config.train.epochs,
                    config.train.batch_size,
                    config.train.device,
                    config.train.optimizer,
                    config.visualization.batching.strategy,
                    config.train.early_stopping.enabled,
                )
                history = train_visualization_tail(
                    tail,
                    r_all[train_idx],
                    r_all[val_idx],
                    train_config,
                    train_family_ids=family_ids[train_idx] if use_family else None,
                    val_family_ids=family_ids[val_idx] if use_family else None,
                    train_spacegroup_ids=(
                        spacegroup_ids[train_idx] if use_spacegroup else None
                    ),
                    val_spacegroup_ids=(
                        spacegroup_ids[val_idx] if use_spacegroup else None
                    ),
                    lambda_family=config.visualization.lambda_family,
                    lambda_spacegroup=config.visualization.lambda_spacegroup,
                    lambda_norm=config.visualization.lambda_norm,
                    batching_strategy=config.visualization.batching.strategy,
                    batching_family_ids=(
                        family_ids[train_idx] if balanced_batching else None
                    ),
                    batching_spacegroup_ids=(
                        spacegroup_ids[train_idx] if balanced_batching else None
                    ),
                    batching_P=config.visualization.batching.balanced_params.P,
                    batching_K=config.visualization.batching.balanced_params.K,
                    batching_S=config.visualization.batching.balanced_params.S,
                )

            actual_epochs = len(history["train_loss"])
            logger.info(
                "Tail training complete: %d/%d epochs, final train_loss=%.4f val_loss=%.4f",
                actual_epochs,
                config.train.epochs,
                history["train_loss"][-1],
                history["val_loss"][-1],
            )
            _save_loss_history(tail_dir / "loss_history.csv", history)

            with open(tail_dir / "tail_params.msgpack", "wb") as f:
                f.write(serialization.to_bytes(tail.params))

            if config.tail_kind == "classification":
                predictions_payload = dict(
                    material_ids=material_ids_all, split=split_all
                )
                eval_targets = []
                if use_family:
                    family_logits_all = tail.classify_family(r_all)
                    family_probs_all = np.asarray(
                        jax.nn.softmax(family_logits_all, axis=-1)
                    )
                    predictions_payload["family_probs"] = family_probs_all
                    predictions_payload["family_classes"] = np.array(family_classes)
                    predictions_payload["labels"] = labels_all
                    eval_targets.append(
                        ("family", labels_all, family_probs_all, family_classes)
                    )
                    if use_spacegroup:
                        spacegroup_logits_all = tail.classify_spacegroup(r_all)
                        masked_logits_all = apply_family_mask(
                            spacegroup_logits_all,
                            family_probs_all,
                            jnp.asarray(family_spacegroup_mask),
                        )
                        spacegroup_probs_all = np.asarray(
                            jax.nn.softmax(masked_logits_all, axis=-1)
                        )
                        predictions_payload["spacegroup_probs"] = spacegroup_probs_all
                        predictions_payload["spacegroup_classes"] = np.array(
                            spacegroup_classes
                        )
                        predictions_payload["spacegroups"] = spacegroups_all
                        eval_targets.append(
                            (
                                "spacegroup",
                                spacegroups_all,
                                spacegroup_probs_all,
                                spacegroup_classes,
                            )
                        )
                np.savez(tail_dir / "tail_predictions.npz", **predictions_payload)
                logger.info(
                    "Saved classification predictions to %s",
                    tail_dir / "tail_predictions.npz",
                )

                # A confusion matrix, a per-class precision/recall/F1 bar chart,
                # and a calibration/reliability diagram, on both the train and
                # val splits, for every active label level -- 6 PNGs (family
                # only) or 12 (family + spacegroup).
                _classifier_eval_plots(
                    tail_dir, eval_targets, {"train": train_idx, "val": val_idx}
                )
            else:
                z_all = np.asarray(tail.project(r_all))
                np.savez(
                    tail_dir / "tail_embeddings.npz",
                    embeddings=z_all,
                    labels=labels_all,
                    material_ids=material_ids_all,
                    spacegroups=spacegroups_all,
                    split=split_all,
                )
                logger.info(
                    "Saved visualization embeddings to %s",
                    tail_dir / "tail_embeddings.npz",
                )

                plot_fn = (
                    plot_reduced_space
                    if config.visualization.viz_dim == 2
                    else plot_reduced_space_3d
                )
                family_plot_path = tail_dir / "viz_plot_family.png"
                plot_fn(
                    z_all,
                    labels_all.tolist(),
                    title=f"{tail_dir.name} ({run_dir.name})",
                    save_path=str(family_plot_path),
                )
                logger.info(
                    "Saved family-colored visualization plot to %s", family_plot_path
                )

                if use_spacegroup:
                    spacegroup_plot_path = tail_dir / "viz_plot_spacegroup.png"
                    plot_fn(
                        z_all,
                        [str(sg) for sg in spacegroups_all.tolist()],
                        title=f"{tail_dir.name} ({run_dir.name}) -- spacegroup",
                        save_path=str(spacegroup_plot_path),
                    )
                    logger.info(
                        "Saved spacegroup-colored visualization plot to %s",
                        spacegroup_plot_path,
                    )

            logger.info("Tail artifacts saved to %s", tail_dir)
    finally:
        logger.removeHandler(file_handler)
        file_handler.close()

    return tail_dir
