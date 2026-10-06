"""
Phase 2 pipeline entry point: freeze an already-trained run's body and train
exactly one tail (classification, visualization, or hierarchical_supcon) on
its saved representations -- see ``dim_red.supcon.tail_training`` for the
actual training loops (reused as-is regardless of which body produced the
representations, since neither training loop ever touches the body itself),
and ``dim_red.pipeline.inference`` for ``load_run_embeddings``, the
lightweight loader used here: this module only ever needs a run's
``config.yaml``/``embeddings.npz``, never the reconstructed body itself (no
forward pass, no encoding), so it deliberately does *not* use
``load_trained_run`` (which would also reconstruct the model, resolve
species, and compute/recompute standardization stats -- all wasted work for
training a tail on already-saved representations).

Which ``model_kind``s a given ``tail_kind`` accepts is gated per-tail-kind by
``_TAIL_MODEL_KINDS`` below.

Reuses phase 1's exact train/val split (``embeddings.npz["split"]``) and
representations (``embeddings.npz["embeddings"]``) -- no SOAP recompute ever
(not even the ``load_trained_run`` fallback for pre-existing runs, since
this module never calls it), no body forward pass at all, so this is freely
rerunnable with different tail configs against the same trained body --
including runs whose ``dataset.extxyz``/``model_params.msgpack`` are no
longer present, since neither is required here. The one exception is a
``hierarchical_supcon`` tail (see ``_train_hierarchical_supcon``/
``_compute_native_soap_features``): it recomputes the native features from
the run's ``dataset.extxyz`` (which must still exist for it), so each
per-family expert trains on the native descriptor instead of the body's
(possibly bottlenecked) embedding.

``tail_kind == "hierarchical_supcon"`` is a genuinely two-stage classifier:
one ``ClassificationTail`` predicts family, then one independent,
separately-trained SupCon stack per family ("expert": encoder + projection,
then a classifier and a visualizer on the frozen expert embedding) predicts
spacegroup, trained only on that family's own rows and only over the
spacegroups actually observed within it.
"""

from __future__ import annotations

import csv
import dataclasses
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jax
import numpy as np
import yaml
from ase.io import read as read_atoms
from flax import serialization

from dim_red.analysis.plotting import (
    plot_classification_report,
    plot_confusion_matrix,
    plot_reduced_space,
    plot_reduced_space_3d,
    plot_reliability_diagram,
)
from dim_red.dataset import FeatureDatabase
from dim_red.pipeline.config import (
    RunConfig,
    TailTrainConfig,
    tail_train_config_to_dict,
)
from dim_red.pipeline.inference import load_run_embeddings
from dim_red.soap import compute_soap
from dim_red.supcon.model import SupConEncoder
from dim_red.supcon.tail_training import TailTrainConfig as SupConTailTrainConfig
from dim_red.supcon.tail_training import (
    train_classification_tail,
    train_visualization_tail,
)
from dim_red.supcon.tails import ClassificationTail, ProjectionTail, VisualizationTail
from dim_red.supcon.training import TrainConfig as SupConBodyTrainConfig
from dim_red.supcon.training import training_first_phase
from dim_red.utils import apply_standardization

logger = logging.getLogger("dim_red.pipeline")

# Which model_kinds a given tail_kind may be trained against, keyed by
# config.TailTrainConfig.tail_kind's valid values. "hierarchical_supcon"
# trains a fresh SupCon body per family on the run's native features, which
# only exist for the SupCon-family model kinds (native SOAP for "supcon",
# native MACE embedding for "supcon_mace" -- see
# _compute_native_soap_features/_compute_native_mace_features); cgcnn has
# neither.
_TAIL_MODEL_KINDS: Dict[str, Tuple[str, ...]] = {
    "classification": ("supcon", "cgcnn", "supcon_mace"),
    "visualization": ("supcon", "cgcnn", "supcon_mace"),
    "hierarchical_supcon": ("supcon", "supcon_mace"),
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
    under ``tails/<...>/sg_experts/``). Family labels are always short
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
    ``"hierarchical_supcon"`` tail kinds.
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


def _compute_native_soap_features(
    run_dir: Path,
    run_config: RunConfig,
    embeddings: Dict[str, np.ndarray],
) -> np.ndarray:
    """Recomputes the native (pre-body) standardized SOAP descriptor for
    ``hierarchical_supcon``'s per-family SupCon bodies -- reads ``dataset.extxyz`` back
    (the exact structures/order the body was trained on), recomputes SOAP
    with the run's own ``soap:`` hyperparameters, and standardizes with the
    run's own saved ``feature_mean``/``feature_std`` (the same statistics
    fit during phase 1 -- never refit here, so this lands in exactly the
    representation the body itself was trained on, just without the body's
    own compression).

    Takes the whole ``embeddings`` dict (not pre-extracted
    ``feature_mean``/``feature_std`` arrays) so the ``run_config.soap``
    check below can run -- and raise its clearer error -- before touching
    ``embeddings`` at all: a non-SOAP run (cgcnn) doesn't even have
    those two keys, so extracting them first would surface a confusing
    ``KeyError`` instead.

    Raises:
        ValueError: If ``run_config.model_kind != "supcon"`` -- ``cgcnn``
            builds its own graph features (no SOAP at all), and
            ``supcon_mace``'s saved ``feature_mean``/``feature_std``
            standardize its MACE embedding, not a SOAP descriptor --
            recomputing "SOAP" from ``run_config.soap`` for either would
            silently use an unrelated, default-valued ``SoapConfig`` (every
            ``RunConfig`` carries one regardless of ``model_kind``),
            producing a nonsensical result instead of a clean error.
        FileNotFoundError: If ``run_dir / "dataset.extxyz"`` doesn't exist.
    """
    if run_config.model_kind != "supcon":
        raise ValueError(
            "hierarchical_supcon on this run requires a SOAP-based body "
            f"(model_kind={run_config.model_kind!r} for {run_dir} -- only "
            "'supcon' runs have a native SOAP descriptor to recompute; "
            "cgcnn uses graph features and supcon_mace uses MACE embeddings)"
        )
    feature_mean = embeddings["feature_mean"]
    feature_std = embeddings["feature_std"]
    dataset_path = run_dir / "dataset.extxyz"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"hierarchical_supcon requires {dataset_path} to "
            "still exist (it recomputes SOAP from it) -- this run's raw "
            "structures appear to have been removed"
        )
    atoms_list = read_atoms(str(dataset_path), index=":")
    soap = run_config.soap
    raw_features = compute_soap(
        atoms_list,
        species=soap.species,
        r_cut=soap.r_cut,
        n_max=soap.n_max,
        l_max=soap.l_max,
        sigma=soap.sigma,
        element_agnostic=soap.element_agnostic,
        average="outer",
    )
    return apply_standardization(raw_features, feature_mean, feature_std)


def _compute_native_mace_features(
    run_dir: Path,
    run_config: RunConfig,
    embeddings: Dict[str, np.ndarray],
) -> np.ndarray:
    """Recomputes the native (pre-body) standardized MACE embedding for
    ``_train_hierarchical_supcon``'s ``model_kind == "supcon_mace"`` branch --
    the exact counterpart of ``_compute_native_soap_features`` for a body
    trained on MACE features instead of SOAP ones. Reads ``dataset.extxyz``
    back (the exact structures/order the body was trained on), runs a fresh
    frozen ``dim_red.mace.model.MaceEncoder`` forward pass with the run's own
    ``mace:`` hyperparameters (the same featurization
    ``dim_red.pipeline.dataset_cache._compute_mace_and_standardize`` used at
    training time), and standardizes with the run's own saved
    ``feature_mean``/``feature_std`` (never refit here) -- lands in exactly
    the representation the ``supcon_mace`` body's encoder itself was trained
    on, just without the body's own compression, mirroring why
    ``_compute_native_soap_features`` recomputes SOAP instead of reusing the
    body's own (possibly bottlenecked) embedding for a plain ``supcon`` run.

    Raises:
        ValueError: If ``run_config.model_kind != "supcon_mace"``.
        FileNotFoundError: If ``run_dir / "dataset.extxyz"`` doesn't exist.
    """
    if run_config.model_kind != "supcon_mace":
        raise ValueError(
            "_compute_native_mace_features requires model_kind='supcon_mace' "
            f"(got {run_config.model_kind!r} for {run_dir})"
        )
    feature_mean = embeddings["feature_mean"]
    feature_std = embeddings["feature_std"]
    dataset_path = run_dir / "dataset.extxyz"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"hierarchical_supcon on a supcon_mace run requires {dataset_path} "
            "to still exist (it recomputes the native MACE embedding from it) "
            "-- this run's raw structures appear to have been removed"
        )
    atoms_list = read_atoms(str(dataset_path), index=":")

    # Imported here, not at module top, so this module stays importable
    # without mace_jax installed unless a supcon_mace run's hierarchical_supcon
    # tail is actually trained -- same lazy-heavy-dependency convention as
    # dim_red.pipeline.single_run/dataset_cache's own MaceEncoder imports.
    from dim_red.mace.model import MaceEncoder

    encoder = MaceEncoder(**run_config.mace.mace_kwargs())
    raw_features = np.asarray(encoder.encode(atoms_list))
    return apply_standardization(raw_features, feature_mean, feature_std)


def _train_hierarchical_supcon(
    config: TailTrainConfig,
    tail_dir: Path,
    run_dir: Path,
    loaded,
) -> None:
    """Train a ``hierarchical_supcon`` tail: stage 1 classifies family on the
    frozen body's embedding; stage 2 trains, per family, a SupCon body
    ("SupCon SG") + classifier + visualizer on that family's own native
    features, mirroring the family-level body/classifier/visualizer pattern
    one level down -- see ``dim_red.pipeline.config.HierarchicalSupconTailConfig``
    for the full rationale.

    The per-family bodies train on the run's *native* features: native SOAP
    for ``model_kind == "supcon"`` (``_compute_native_soap_features``), the
    native MACE embedding for ``"supcon_mace"``
    (``_compute_native_mace_features``) -- the same substitution that run's
    own family-level body already makes.

    Predictions are assembled into a ``tail_predictions.npz`` schema every
    downstream reader (``dim_red.pipeline.compare.
    hierarchical_accuracies_from_npz``, ``dim_red.pipeline.benchmark``, the
    confusion-matrix/classification-report/calibration plots) understands.
    """
    hs = config.hierarchical_supcon
    is_supcon_mace = loaded.config.model_kind == "supcon_mace"
    r_all = loaded.embeddings["embeddings"]
    labels_all = loaded.embeddings["labels"]
    spacegroups_all = loaded.embeddings["spacegroups"]
    material_ids_all = loaded.embeddings["material_ids"]
    split_all = loaded.embeddings["split"]
    train_idx = split_all == "train"
    val_idx = split_all == "val"

    if is_supcon_mace:
        logger.info(
            "Recomputing native MACE features from %s for hierarchical_supcon's "
            "per-family SupCon SG bodies (model_kind=supcon_mace)",
            run_dir / "dataset.extxyz",
        )
        r_soap_all = _compute_native_mace_features(
            run_dir, loaded.config, loaded.embeddings
        )
        soap_input_dim = r_soap_all.shape[1]
        logger.info("Recomputed native MACE features: shape=%s", r_soap_all.shape)
    else:
        logger.info(
            "Recomputing native SOAP features from %s for hierarchical_supcon's "
            "per-family SupCon SG bodies",
            run_dir / "dataset.extxyz",
        )
        r_soap_all = _compute_native_soap_features(
            run_dir, loaded.config, loaded.embeddings
        )
        soap_input_dim = r_soap_all.shape[1]
        logger.info("Recomputed native SOAP features: shape=%s", r_soap_all.shape)

    # Representation width fed to the per-family classifier/visualizer: the
    # SG body's own trained latent_dim.
    sg_repr_dim = hs.sg_latent_dim

    family_classes, family_ids = _build_vocab_ids(labels_all.tolist())
    logger.info("%d family classes: %s", len(family_classes), family_classes)
    spacegroup_classes, _ = _build_vocab_ids(spacegroups_all.tolist())
    spacegroup_class_to_col = {v: i for i, v in enumerate(spacegroup_classes)}
    n_spacegroup = len(spacegroup_classes)
    logger.info("%d spacegroup classes observed", n_spacegroup)

    logger.info(
        "Training hierarchical_supcon tail: head_hidden_dim=%d "
        "sg_encoder_hidden_dim=%s sg_latent_dim=%d sg_tau=%.3f "
        "sg_distance=%s min_samples_per_expert=%d epochs=%d batch_size=%d "
        "device=%s early_stopping=%s",
        hs.head_hidden_dim,
        hs.sg_encoder_hidden_dim,
        hs.sg_latent_dim,
        hs.sg_tau,
        hs.sg_distance,
        hs.min_samples_per_expert,
        config.train.epochs,
        config.train.batch_size,
        config.train.device,
        config.train.early_stopping.enabled,
    )
    tail_train_config = SupConTailTrainConfig(
        epochs=config.train.epochs,
        batch_size=config.train.batch_size,
        learning_rate=config.train.learning_rate,
        seed=config.seed,
        device=config.train.device,
        early_stopping=config.train.early_stopping.enabled,
        early_stopping_patience=config.train.early_stopping.patience,
        early_stopping_min_delta=config.train.early_stopping.min_delta,
        early_stopping_restore_best=config.train.early_stopping.restore_best_weights,
    )
    # The per-family SG visualizer's own contrastive loss settings -- same
    # training mechanics as tail_train_config, only tau/distance differ (the
    # classifiers above never read tau/distance, so they keep the base config).
    sg_viz_train_config = dataclasses.replace(
        tail_train_config,
        tau=hs.sg_visualization_tau,
        distance=hs.sg_visualization_distance,
    )
    sg_body_train_config = SupConBodyTrainConfig(
        epochs=config.train.epochs,
        batch_size=config.train.batch_size,
        learning_rate=config.train.learning_rate,
        tau=hs.sg_tau,
        distance=hs.sg_distance,
        seed=config.seed,
        device=config.train.device,
        early_stopping=config.train.early_stopping.enabled,
        early_stopping_patience=config.train.early_stopping.patience,
        early_stopping_min_delta=config.train.early_stopping.min_delta,
        early_stopping_restore_best=config.train.early_stopping.restore_best_weights,
    )

    # --- Stage 1: family ------------------------------------------------------
    family_dir = tail_dir / "family"
    family_dir.mkdir(parents=True, exist_ok=True)
    family_tail = ClassificationTail(
        input_dim=r_all.shape[1],
        hidden_dim=hs.head_hidden_dim,
        n_classes=len(family_classes),
        seed=config.seed,
    )
    family_history = train_classification_tail(
        family_tail,
        r_all[train_idx],
        r_all[val_idx],
        tail_train_config,
        train_labels=family_ids[train_idx],
        val_labels=family_ids[val_idx],
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

    family_probs_all = np.asarray(jax.nn.softmax(family_tail.classify(r_all), axis=-1))
    family_pred_ids = family_probs_all.argmax(axis=1)

    # --- Stage 2: SupCon SG body + classifier + visualizer, per family -------
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

        if n_train_k < hs.min_samples_per_expert or len(local_classes) < 2:
            # Not enough data (or not enough spacegroup variety) for a
            # dedicated SG expert -- always predict this family's single most
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
                hs.min_samples_per_expert,
                majority_value,
            )
            continue

        # A family with no validation rows of its own reuses its training
        # rows for "validation" too.
        expert_val_mask = (
            family_mask_val if family_mask_val.any() else family_mask_train
        )
        sg_dir = tail_dir / "sg_experts" / _sanitize_family_dirname(family_name)
        sg_dir.mkdir(parents=True, exist_ok=True)

        # 1. SupCon SG: a fresh body+projection tail trained from scratch
        # on this family's native subset -- SOAP for a plain "supcon"
        # run, MACE for "supcon_mace" (r_soap_all/soap_input_dim were
        # already resolved to whichever one applies, above) --
        # contrasting on local spacegroup id (the spacegroup slot
        # directly -- family is constant within this subset, so that
        # term is left inactive).
        sg_body = SupConEncoder(
            input_dim=soap_input_dim,
            encoder_hidden_dim=hs.sg_encoder_hidden_dim,
            latent_dim=hs.sg_latent_dim,
            seed=config.seed,
        )
        sg_projection = ProjectionTail(
            input_dim=hs.sg_latent_dim,
            hidden_dim=hs.sg_projection_hidden_dim or [hs.sg_latent_dim],
            projection_dim=hs.sg_projection_dim,
            seed=config.seed,
        )
        sg_body_history = training_first_phase(
            sg_body,
            sg_projection,
            FeatureDatabase.from_array(r_soap_all[family_mask_train]),
            FeatureDatabase.from_array(r_soap_all[expert_val_mask]),
            sg_body_train_config,
            train_spacegroup_ids=local_ids_full[family_mask_train],
            val_spacegroup_ids=local_ids_full[expert_val_mask],
            lambda_family=0.0,
            lambda_spacegroup=1.0,
            lambda_norm=hs.sg_lambda_norm,
        )
        logger.info(
            "Family %r: trained SupCon SG body on %d training rows, %d local "
            "spacegroup classes, %d/%d epochs, final train_loss=%.4f "
            "val_loss=%.4f",
            family_name,
            n_train_k,
            len(local_classes),
            len(sg_body_history["train_loss"]),
            config.train.epochs,
            sg_body_history["train_loss"][-1],
            sg_body_history["val_loss"][-1],
        )
        _save_loss_history(sg_dir / "sg_body_loss_history.csv", sg_body_history)
        with open(sg_dir / "sg_body_params.msgpack", "wb") as f:
            f.write(serialization.to_bytes(sg_body.params))
        with open(sg_dir / "sg_projection_params.msgpack", "wb") as f:
            f.write(serialization.to_bytes(sg_projection.params))
        with open(sg_dir / "local_spacegroup_classes.yaml", "w") as f:
            yaml.safe_dump(
                {"local_spacegroup_classes": [int(c) for c in local_classes]}, f
            )

        # Frozen SG embedding for every row of this family (train+val): the
        # just-trained SG body's own encode().
        r_sg_family_all = np.asarray(sg_body.encode(r_soap_all[family_mask_all]))
        family_positions = np.flatnonzero(family_mask_all)
        pos_lookup = {row: i for i, row in enumerate(family_positions)}
        train_pos = np.array([pos_lookup[i] for i in np.flatnonzero(family_mask_train)])
        val_pos = np.array([pos_lookup[i] for i in np.flatnonzero(expert_val_mask)])
        local_sg_ids_family = local_ids_full[family_mask_all]

        # 2. Classifier on top of the frozen SG embedding.
        sg_classifier = ClassificationTail(
            input_dim=sg_repr_dim,
            hidden_dim=hs.sg_classifier_hidden_dim,
            n_classes=len(local_classes),
            seed=config.seed,
        )
        sg_classifier_history = train_classification_tail(
            sg_classifier,
            r_sg_family_all[train_pos],
            r_sg_family_all[val_pos],
            tail_train_config,
            train_labels=local_sg_ids_family[train_pos],
            val_labels=local_sg_ids_family[val_pos],
        )
        _save_loss_history(
            sg_dir / "classifier_loss_history.csv", sg_classifier_history
        )
        with open(sg_dir / "classifier_tail_params.msgpack", "wb") as f:
            f.write(serialization.to_bytes(sg_classifier.params))

        # 3. Visualization tail on top of the same frozen SG embedding. Uses
        # this family's own tuned hidden_dim when present in
        # sg_visualization_hidden_dim_by_family (round 16's per-family
        # tuning result), else falls back to the scalar sg_visualization_hidden_dim.
        sg_viz_hidden_dim = hs.sg_visualization_hidden_dim_by_family.get(
            family_name, hs.sg_visualization_hidden_dim
        )
        sg_viz = VisualizationTail(
            input_dim=sg_repr_dim,
            hidden_dim=sg_viz_hidden_dim,
            output_dim=2,
            seed=config.seed,
        )
        sg_viz_history = train_visualization_tail(
            sg_viz,
            r_sg_family_all[train_pos],
            r_sg_family_all[val_pos],
            sg_viz_train_config,
            train_spacegroup_ids=local_sg_ids_family[train_pos],
            val_spacegroup_ids=local_sg_ids_family[val_pos],
            lambda_family=0.0,
            lambda_spacegroup=1.0,
            lambda_norm=hs.sg_visualization_lambda_norm,
        )
        _save_loss_history(sg_dir / "visualization_loss_history.csv", sg_viz_history)
        with open(sg_dir / "visualization_tail_params.msgpack", "wb") as f:
            f.write(serialization.to_bytes(sg_viz.params))
        z_family = np.asarray(sg_viz.project(r_sg_family_all))
        family_split = np.where(family_mask_train[family_positions], "train", "val")
        np.savez(
            sg_dir / "visualization_embeddings.npz",
            embeddings=z_family,
            spacegroups=local_spacegroups,
            material_ids=material_ids_all[family_positions],
            split=family_split,
        )
        plot_reduced_space(
            z_family,
            [str(sg) for sg in local_spacegroups.tolist()],
            title=f"{family_name} SupCon SG visualization ({run_dir.name})",
            save_path=str(sg_dir / "visualization_plot_spacegroup.png"),
            legend=False,
        )
        logger.info(
            "Saved family %r SupCon SG visualization plot to %s",
            family_name,
            sg_dir / "visualization_plot_spacegroup.png",
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
            "Family %r: trained SG classifier+visualizer, %d/%d epochs, "
            "final train_loss=%.4f val_loss=%.4f",
            family_name,
            len(sg_classifier_history["train_loss"]),
            config.train.epochs,
            sg_classifier_history["train_loss"][-1],
            sg_classifier_history["val_loss"][-1],
        )

        cols = [spacegroup_class_to_col[c] for c in local_classes]
        if predicted_mask.any():
            r_sg_predicted = np.asarray(sg_body.encode(r_soap_all[predicted_mask]))
            local_probs = np.asarray(
                jax.nn.softmax(sg_classifier.classify(r_sg_predicted), axis=-1)
            )
            spacegroup_probs_all[np.ix_(predicted_mask, cols)] = local_probs
        if true_mask.any():
            local_probs_oracle = np.asarray(
                jax.nn.softmax(sg_classifier.classify(r_sg_family_all), axis=-1)
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
        "Saved hierarchical_supcon classification predictions to %s",
        tail_dir / "tail_predictions.npz",
    )

    eval_targets = [
        ("family", labels_all, family_probs_all, family_classes),
        ("spacegroup", spacegroups_all, spacegroup_probs_all, spacegroup_classes),
    ]
    _classifier_eval_plots(tail_dir, eval_targets, {"train": train_idx, "val": val_idx})


def train_tail(config: TailTrainConfig) -> Path:
    """Freeze a completed run's body and train exactly one tail
    (classification, visualization, or hierarchical_supcon) on its
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
        ``tail_embeddings.npz`` plus plot(s); or for
        ``tail_kind="hierarchical_supcon"``: ``family/`` (stage-1 tail),
        ``sg_experts/<family>/`` (one per trained stage-2 expert),
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

        if config.tail_kind == "hierarchical_supcon":
            with open(tail_dir / "tail_config.yaml", "w") as f:
                yaml.safe_dump(tail_train_config_to_dict(config), f, sort_keys=False)
            _train_hierarchical_supcon(config, tail_dir, run_dir, loaded)
            logger.info("Tail artifacts saved to %s", tail_dir)
        else:
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
                logger.info(
                    "%d family classes: %s", len(family_classes), family_classes
                )
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
                predictions_payload = dict(
                    material_ids=material_ids_all, split=split_all
                )
                eval_targets = []
                family_logits_all = tail.classify(r_all)
                family_probs_all = np.asarray(
                    jax.nn.softmax(family_logits_all, axis=-1)
                )
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
