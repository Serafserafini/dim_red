"""
Phase 2 pipeline entry point: freeze an already-trained ``model: supcon``
run's body and train exactly one tail (classification or visualization) on
its saved representations -- see ``dim_red.supcon.tail_training`` for the
actual training loops, and ``dim_red.pipeline.inference`` for
``load_run_embeddings``, the lightweight loader used here: this module only
ever needs a run's ``config.yaml``/``embeddings.npz``, never the
reconstructed body itself (no forward pass, no encoding), so it deliberately
does *not* use ``load_trained_run`` (which would also reconstruct the model,
resolve species, and compute/recompute standardization stats -- all wasted
work for training a tail on already-saved representations).

Reuses phase 1's exact train/val split (``embeddings.npz["split"]``) and
representations (``embeddings.npz["embeddings"]``) -- no SOAP recompute ever
(not even the ``load_trained_run`` fallback for pre-existing runs, since
this module never calls it), no body forward pass at all, so this is freely
rerunnable with different tail configs against the same trained body --
including runs whose ``dataset.extxyz``/``model_params.msgpack`` are no
longer present, since neither is required here.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Tuple

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


def train_tail(config: TailTrainConfig) -> Path:
    """Freeze a completed ``model: supcon`` run's body and train exactly one
    tail (classification or visualization) on its already-saved
    representations, saving the tail's own artifacts to
    ``<run_dir>/tails/<output_subdir or tail_kind>/``.

    Args:
        config: Which run/tail/hyperparameters to train -- see
            ``dim_red.pipeline.config.TailTrainConfig``.

    Returns:
        The tail's output directory (``run.log`` -- every log line emitted
        from here on, mirroring ``dim_red.pipeline.single_run.run_single``'s
        own ``run.log`` -- ``tail_config.yaml``, ``tail_params.msgpack``,
        ``loss_history.csv``, plus either: ``tail_predictions.npz``
        (predicted probabilities, class vocabularies, and the true labels,
        for a classification tail) and a confusion matrix/precision-recall-F1
        bar chart/calibration diagram per active label level and per
        train/val split (``confusion_matrix_<level>_<split>.png``,
        ``classification_report_<level>_<split>.png``,
        ``calibration_<level>_<split>.png``); or ``tail_embeddings.npz``/
        plot(s) for a visualization tail).

    Raises:
        ValueError: If ``config.run_dir`` is unset, or wasn't produced by a
            ``model_kind == "supcon"`` run.
    """
    if config.run_dir is None:
        raise ValueError(
            "config.run_dir is unset -- pass a run directory on the command "
            "line (dimred-train-tail <config> <run_dir>) or set run_dir "
            "directly in the TailTrainConfig/YAML"
        )
    run_dir = Path(config.run_dir)
    loaded = load_run_embeddings(run_dir)
    if loaded.config.model_kind != "supcon":
        raise ValueError(
            f"{run_dir} is a model_kind={loaded.config.model_kind!r} run -- "
            "tail training requires a completed model_kind='supcon' run "
            "(only that body has no classification/visualization capability "
            "of its own to begin with)"
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
            logger.info("%d family classes: %s", len(family_classes), family_classes)
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
                config.visualization.tau if config.tail_kind == "visualization" else 0.1
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
            predictions_payload = dict(material_ids=material_ids_all, split=split_all)
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
            split_masks = {"train": train_idx, "val": val_idx}
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
                "Saved visualization embeddings to %s", tail_dir / "tail_embeddings.npz"
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
