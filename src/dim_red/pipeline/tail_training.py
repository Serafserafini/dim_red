"""
Phase 2 pipeline entry point: freeze an already-trained cgcnn run's body and
train exactly one tail (classification or visualization) on its saved
representations -- see ``dim_red.supcon.tail_training`` for the actual
training loops (reused as-is regardless of which body produced the
representations, since neither training loop ever touches the body itself),
and ``dim_red.pipeline.inference`` for ``load_run_embeddings``, the
lightweight loader used here: this module only ever needs a run's
``config.yaml``/``embeddings.npz``, never the reconstructed body itself (no
forward pass, no encoding), so it deliberately does *not* use
``load_trained_run``.

Which ``model_kind``s a given ``tail_kind`` accepts is gated per-tail-kind by
``_TAIL_MODEL_KINDS`` below: only ``cgcnn`` runs (written by
``dim_red.pipeline.single_run.run_single``). supcon/supcon_mace stacks train
their heads through ``dim_red.pipeline.full_stack.FullStack`` instead.

Reuses phase 1's exact train/val split (``embeddings.npz["split"]``) and
representations (``embeddings.npz["embeddings"]``) -- no body forward pass at
all, so this is freely rerunnable with different tail configs against the
same trained body -- including runs whose ``dataset.extxyz``/
``model_params.msgpack`` are no longer present, since neither is required
here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import jax
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
from dim_red.pipeline._common import (
    _build_vocab_ids,
    _make_unique_run_dir,
    _save_loss_history,
)
from dim_red.pipeline.config import TailTrainConfig, tail_train_config_to_dict
from dim_red.pipeline.inference import load_run_embeddings
from dim_red.supcon.tail_training import TailTrainConfig as SupConTailTrainConfig
from dim_red.supcon.tail_training import (
    train_classification_tail,
    train_visualization_tail,
)
from dim_red.supcon.tails import ClassificationTail, VisualizationTail

logger = logging.getLogger("dim_red.pipeline")

# Which model_kinds a given tail_kind may be trained against, keyed by
# config.TailTrainConfig.tail_kind's valid values. Only cgcnn runs remain
# here; supcon/supcon_mace heads are trained by FullStack.
_TAIL_MODEL_KINDS: Dict[str, Tuple[str, ...]] = {
    "classification": ("cgcnn",),
    "visualization": ("cgcnn",),
}


def _classifier_eval_plots(
    tail_dir: Path,
    eval_targets: List[Tuple[str, np.ndarray, np.ndarray, List]],
    split_masks: Dict[str, np.ndarray],
) -> None:
    """A confusion matrix, a per-class precision/recall/F1 bar chart, and a
    calibration/reliability diagram, on every named split, for every
    ``(level, y_true_all, y_probs_all, class_names)`` entry in
    ``eval_targets`` -- used by the ``"classification"`` tail kind.
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


def train_tail(config: TailTrainConfig) -> Path:
    """Freeze a completed run's body and train exactly one tail
    (classification or visualization) on its
    already-saved representations, saving the tail's own artifacts to
    ``<run_dir>/tails/<output_subdir or tail_kind>/``. Which ``model_kind``s
    are accepted depends on ``config.tail_kind`` -- see ``_TAIL_MODEL_KINDS``.

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
        ``tail_embeddings.npz`` plus plot(s).

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
    allowed_model_kinds = _TAIL_MODEL_KINDS.get(config.tail_kind, ())
    if loaded.config.model_kind not in allowed_model_kinds:
        raise ValueError(
            f"{run_dir} is a model_kind={loaded.config.model_kind!r} run -- "
            f"tail_kind={config.tail_kind!r} requires a completed run whose "
            f"model_kind is one of {allowed_model_kinds}"
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
            mode = "family_only"
            use_family = True
            use_spacegroup = False
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
        if need_family_ids:
            family_classes, family_ids = _build_vocab_ids(labels_all.tolist())
            logger.info("%d family classes: %s", len(family_classes), family_classes)
        if need_spacegroup_ids:
            spacegroup_classes, spacegroup_ids = _build_vocab_ids(
                spacegroups_all.tolist()
            )
            logger.info("%d spacegroup classes observed", len(spacegroup_classes))

        with open(tail_dir / "tail_config.yaml", "w") as f:
            yaml.safe_dump(tail_train_config_to_dict(config), f, sort_keys=False)

        train_config = SupConTailTrainConfig(
            epochs=config.train.epochs,
            batch_size=config.train.batch_size,
            learning_rate=config.train.learning_rate,
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
                n_classes=len(family_classes),
                seed=config.seed,
            )
            logger.info(
                "Training classification tail: head_hidden_dim=%s epochs=%d "
                "batch_size=%d device=%s early_stopping=%s",
                config.classification.head_hidden_dim,
                config.train.epochs,
                config.train.batch_size,
                config.train.device,
                config.train.early_stopping.enabled,
            )
            history = train_classification_tail(
                tail,
                r_all[train_idx],
                r_all[val_idx],
                train_config,
                train_labels=family_ids[train_idx],
                val_labels=family_ids[val_idx],
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
                "epochs=%d batch_size=%d device=%s batching=%s "
                "early_stopping=%s",
                config.visualization.viz_dim,
                mode,
                config.visualization.tau,
                config.visualization.distance,
                config.train.epochs,
                config.train.batch_size,
                config.train.device,
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
            family_logits_all = tail.classify(r_all)
            family_probs_all = np.asarray(jax.nn.softmax(family_logits_all, axis=-1))
            predictions_payload["family_probs"] = family_probs_all
            predictions_payload["family_classes"] = np.array(family_classes)
            predictions_payload["labels"] = labels_all
            eval_targets.append(
                ("family", labels_all, family_probs_all, family_classes)
            )
            np.savez(tail_dir / "tail_predictions.npz", **predictions_payload)
            logger.info(
                "Saved classification predictions to %s",
                tail_dir / "tail_predictions.npz",
            )

            # A confusion matrix, a per-class precision/recall/F1 bar chart,
            # and a calibration/reliability diagram, on both the train and
            # val splits -- 6 PNGs.
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
                legend=False,
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
                    legend=False,
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
