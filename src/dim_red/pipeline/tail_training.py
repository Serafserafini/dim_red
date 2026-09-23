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
longer present, since neither is required here. The one exception is a
hierarchical tail with ``hierarchical.expert_input: "soap"`` (see
``_train_hierarchical_tail``/``_compute_native_soap_features``): that
option does recompute SOAP from the run's ``dataset.extxyz`` (which must
still exist for it), specifically so stage 2's experts can train on the
native descriptor instead of the body's (possibly bottlenecked) embedding.

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
from ase.io import read as read_atoms
from flax import serialization

from dim_red.analysis.plotting import (
    plot_classification_report,
    plot_confusion_matrix,
    plot_reduced_space,
    plot_reduced_space_3d,
    plot_reliability_diagram,
)
from dim_red.pipeline.config import (
    RunConfig,
    TailTrainConfig,
    tail_train_config_to_dict,
)
from dim_red.pipeline.inference import load_classification_tail, load_run_embeddings
from dim_red.soap import compute_soap
from dim_red.supcon.model import SupConEncoder
from dim_red.supcon.tail_training import TailTrainConfig as SupConTailTrainConfig
from dim_red.supcon.tail_training import (
    train_classification_tail,
    train_visualization_tail,
)
from dim_red.supcon.tails import (
    ClassificationTail,
    ProjectionTail,
    VisualizationTail,
    apply_family_mask,
)
from dim_red.supcon.training import TrainConfig as SupConBodyTrainConfig
from dim_red.supcon.training import train_supcon
from dim_red.utils import apply_standardization
from dim_red.vae.database import VAEDatabase

logger = logging.getLogger("dim_red.pipeline")

# Which model_kinds a given tail_kind may be trained against -- see the
# module docstring above for the reasoning. Keyed by config.TailTrainConfig
# .tail_kind's own four valid values. "hierarchical_visualization" is
# restricted to "supcon" only (not just "supcon"/"cgcnn"/"mace" like
# "hierarchical" itself): it always recomputes native SOAP when the
# referenced hierarchical tail used expert_input="soap" (the default), and
# only a SOAP-based (supcon) body has that descriptor to recompute at all.
# "hierarchical_supcon" also allows "mace" (not "cgcnn"): its per-family
# stage-2 step normally trains a fresh SupCon body on native SOAP, which
# only a SOAP-based body has -- but model_kind: mace never trains anything
# at all (a frozen foundation-model forward pass stands in for a from-
# scratch encoder+projection), so _train_hierarchical_supcon skips that
# per-family training step for mace and reuses the run's own frozen
# embedding directly as each family's SG representation instead, same
# substitution the family-level body already makes. cgcnn isn't included
# here: it neither has SOAP to recompute (like mace) nor is a frozen body
# (unlike mace, it trains from scratch), so neither branch applies to it.
_TAIL_MODEL_KINDS: Dict[str, Tuple[str, ...]] = {
    "classification": ("supcon", "cgcnn", "mace"),
    "visualization": ("supcon", "cgcnn", "mace", "vae", "autoencoder"),
    "hierarchical": ("supcon", "cgcnn", "mace"),
    "hierarchical_visualization": ("supcon",),
    "hierarchical_supcon": ("supcon", "mace"),
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


def _compute_native_soap_features(
    run_dir: Path,
    run_config: RunConfig,
    embeddings: Dict[str, np.ndarray],
) -> np.ndarray:
    """Recomputes the native (pre-body) standardized SOAP descriptor for
    ``hierarchical.expert_input: "soap"`` -- reads ``dataset.extxyz`` back
    (the exact structures/order the body was trained on), recomputes SOAP
    with the run's own ``soap:`` hyperparameters, and standardizes with the
    run's own saved ``feature_mean``/``feature_std`` (the same statistics
    fit during phase 1 -- never refit here, so this lands in exactly the
    representation the body itself was trained on, just without the body's
    own compression).

    Takes the whole ``embeddings`` dict (not pre-extracted
    ``feature_mean``/``feature_std`` arrays) so the ``run_config.soap``
    check below can run -- and raise its clearer error -- before touching
    ``embeddings`` at all: a non-SOAP run (cgcnn/mace) doesn't even have
    those two keys, so extracting them first would surface a confusing
    ``KeyError`` instead.

    Raises:
        ValueError: If ``run_config.model_kind != "supcon"`` -- ``cgcnn``
            builds its own graph features (no SOAP at all), and ``mace``'s
            saved ``feature_mean``/``feature_std`` standardize its frozen
            foundation-model embedding, not a SOAP descriptor -- recomputing
            "SOAP" from ``run_config.soap`` for either would silently use
            an unrelated, default-valued ``SoapConfig`` (every ``RunConfig``
            carries one regardless of ``model_kind``, whether or not that
            model_kind's own pipeline ever reads it), producing a
            nonsensical result instead of a clean error.
        FileNotFoundError: If ``run_dir / "dataset.extxyz"`` doesn't exist.
    """
    if run_config.model_kind != "supcon":
        raise ValueError(
            "hierarchical.expert_input='soap' requires a SOAP-based body "
            f"(model_kind={run_config.model_kind!r} for {run_dir} -- only "
            "'supcon' runs have a native SOAP descriptor to recompute; "
            "cgcnn uses graph features and mace uses a frozen foundation-"
            "model embedding, neither of which is SOAP)"
        )
    feature_mean = embeddings["feature_mean"]
    feature_std = embeddings["feature_std"]
    dataset_path = run_dir / "dataset.extxyz"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"hierarchical.expert_input='soap' requires {dataset_path} to "
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
        normalize_distances=soap.normalize_distances,
    )
    return apply_standardization(raw_features, feature_mean, feature_std)


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
    r_expert_all: "np.ndarray | None" = None,
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

    Args:
        r_expert_all: Representation stage-2 experts train/predict on --
            ``None`` (default) reuses ``r_all`` (the body embeddings,
            current/original behavior); pass a separate array (e.g. native
            SOAP features, see ``_compute_native_soap_features``) to train
            experts on a different representation than stage 1's family
            classifier, which always uses ``r_all`` regardless.
    """
    hierarchical = config.hierarchical
    input_dim = r_all.shape[1]
    if r_expert_all is None:
        r_expert_all = r_all
    expert_input_dim = r_expert_all.shape[1]
    expert_hidden_dim = (
        hierarchical.expert_head_hidden_dim
        if hierarchical.expert_head_hidden_dim is not None
        else hierarchical.head_hidden_dim
    )

    family_classes, family_ids = _build_vocab_ids(labels_all.tolist())
    logger.info("%d family classes: %s", len(family_classes), family_classes)
    spacegroup_classes, _ = _build_vocab_ids(spacegroups_all.tolist())
    spacegroup_class_to_col = {v: i for i, v in enumerate(spacegroup_classes)}
    n_spacegroup = len(spacegroup_classes)
    logger.info("%d spacegroup classes observed", n_spacegroup)

    logger.info(
        "Training hierarchical tail: head_hidden_dim=%d expert_hidden_dim=%s "
        "min_samples_per_expert=%d expert_input=%s (dim=%d) epochs=%d "
        "batch_size=%d device=%s optimizer=%s early_stopping=%s",
        hierarchical.head_hidden_dim,
        expert_hidden_dim,
        hierarchical.min_samples_per_expert,
        hierarchical.expert_input,
        expert_input_dim,
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
            input_dim=expert_input_dim,
            hidden_dim=expert_hidden_dim,
            n_family_classes=len(local_classes),
            seed=config.seed,
        )
        expert_history = train_classification_tail(
            expert_tail,
            r_expert_all[family_mask_train],
            r_expert_all[expert_val_mask],
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
                    expert_tail.classify_family(r_expert_all[predicted_mask]), axis=-1
                )
            )
            spacegroup_probs_all[np.ix_(predicted_mask, cols)] = local_probs
        if true_mask.any():
            local_probs_oracle = np.asarray(
                jax.nn.softmax(
                    expert_tail.classify_family(r_expert_all[true_mask]), axis=-1
                )
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


def _train_hierarchical_supcon(
    config: TailTrainConfig,
    tail_dir: Path,
    run_dir: Path,
    loaded,
) -> None:
    """Experimental variant of ``_train_hierarchical_tail``: stage 1
    (family) is identical, but stage 2's per-family expert is itself
    restructured as a SupCon body ("SupCon SG") + classifier + visualizer,
    mirroring the family-level body/classifier/visualizer pattern one level
    down -- see ``dim_red.pipeline.config.HierarchicalSupconTailConfig`` for
    the full rationale.

    For ``model_kind == "mace"`` there is no per-family body/projection
    training step at all: a frozen MACE embedding IS the "encoder +
    projection" (this is exactly the same substitution the family-level
    body makes -- ``model_kind: mace`` never trains anything, an already-
    pretrained foundation-model forward pass stands in for the SupCon
    encoder+projection there too), so stage 2 reuses the run's own frozen
    embedding (``r_all``, restricted to that family's rows) directly as
    each family's representation, skipping straight to the classifier +
    visualizer steps -- no ``sg_body``/``sg_projection``, no native-SOAP
    recompute (a mace run has no SOAP features to recompute at all), and
    ``hs.sg_encoder_hidden_dim``/``sg_latent_dim``/``sg_tau``/``sg_distance``/
    ``sg_lambda_norm``/``sg_projection_dim``/``sg_projection_hidden_dim``
    (all body/projection-training hyperparameters) go unused.

    Predictions are assembled into the exact same ``tail_predictions.npz``
    schema ``"hierarchical"`` already produces, so every existing
    downstream reader (``dim_red.pipeline.compare.
    hierarchical_accuracies_from_npz``, ``dim_red.pipeline.benchmark``, the
    confusion-matrix/classification-report/calibration plots) works
    unmodified.
    """
    hs = config.hierarchical_supcon
    is_mace = loaded.config.model_kind == "mace"
    r_all = loaded.embeddings["embeddings"]
    labels_all = loaded.embeddings["labels"]
    spacegroups_all = loaded.embeddings["spacegroups"]
    material_ids_all = loaded.embeddings["material_ids"]
    split_all = loaded.embeddings["split"]
    train_idx = split_all == "train"
    val_idx = split_all == "val"

    if is_mace:
        # The frozen MACE embedding replaces the per-family SupCon SG body
        # entirely -- see the docstring above. r_soap_all/soap_input_dim
        # below are simply r_all/its own width in this case (never a real
        # SOAP descriptor), kept as the same two names so the per-family
        # loop further down needs no separate mace-vs-supcon branching
        # beyond the body-training block itself.
        r_soap_all = r_all
        soap_input_dim = r_all.shape[1]
        logger.info(
            "model_kind=mace: using the run's own frozen MACE embedding "
            "(dim=%d) directly as each family's SG representation -- no "
            "per-family body/projection training, no SOAP recompute",
            soap_input_dim,
        )
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

    # Representation width fed to the per-family classifier/visualizer --
    # the SG body's own trained latent_dim normally, or the frozen MACE
    # embedding's width when there's no SG body being trained at all.
    sg_repr_dim = soap_input_dim if is_mace else hs.sg_latent_dim

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
        "device=%s optimizer=%s early_stopping=%s",
        hs.head_hidden_dim,
        hs.sg_encoder_hidden_dim,
        hs.sg_latent_dim,
        hs.sg_tau,
        hs.sg_distance,
        hs.min_samples_per_expert,
        config.train.epochs,
        config.train.batch_size,
        config.train.device,
        config.train.optimizer,
        config.train.early_stopping.enabled,
    )
    tail_train_config = SupConTailTrainConfig(
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
    sg_body_train_config = SupConBodyTrainConfig(
        epochs=config.train.epochs,
        batch_size=config.train.batch_size,
        learning_rate=config.train.learning_rate,
        optimizer=config.train.optimizer,
        tau=hs.sg_tau,
        distance=hs.sg_distance,
        seed=config.seed,
        device=config.train.device,
        early_stopping=config.train.early_stopping.enabled,
        early_stopping_patience=config.train.early_stopping.patience,
        early_stopping_min_delta=config.train.early_stopping.min_delta,
        early_stopping_restore_best=config.train.early_stopping.restore_best_weights,
    )

    # --- Stage 1: family (identical to "hierarchical") -----------------------
    family_dir = tail_dir / "family"
    family_dir.mkdir(parents=True, exist_ok=True)
    family_tail = ClassificationTail(
        input_dim=r_all.shape[1],
        hidden_dim=hs.head_hidden_dim,
        n_family_classes=len(family_classes),
        seed=config.seed,
    )
    family_history = train_classification_tail(
        family_tail,
        r_all[train_idx],
        r_all[val_idx],
        tail_train_config,
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
            # Same fallback as "hierarchical": not enough data (or not
            # enough spacegroup variety) for a dedicated SG expert -- always
            # predict this family's single most frequent training-set
            # spacegroup instead.
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
        # rows for "validation" too, same as "hierarchical".
        expert_val_mask = (
            family_mask_val if family_mask_val.any() else family_mask_train
        )
        sg_dir = tail_dir / "sg_experts" / _sanitize_family_dirname(family_name)
        sg_dir.mkdir(parents=True, exist_ok=True)

        sg_body = None
        if is_mace:
            # No body/projection to train: the frozen MACE embedding IS the
            # SG representation for this family, same substitution the
            # family-level body already makes.
            with open(sg_dir / "local_spacegroup_classes.yaml", "w") as f:
                yaml.safe_dump(
                    {"local_spacegroup_classes": [int(c) for c in local_classes]}, f
                )
            logger.info(
                "Family %r: using the frozen MACE embedding directly (no SG "
                "body trained), %d training rows, %d local spacegroup classes",
                family_name,
                n_train_k,
                len(local_classes),
            )
        else:
            # 1. SupCon SG: a fresh body+projection tail trained from scratch
            # on this family's native SOAP subset, contrasting on local
            # spacegroup id (the spacegroup slot directly -- family is
            # constant within this subset, so that term is left inactive).
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
            sg_body_history = train_supcon(
                sg_body,
                sg_projection,
                VAEDatabase.from_array(r_soap_all[family_mask_train]),
                VAEDatabase.from_array(r_soap_all[expert_val_mask]),
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

        # Frozen SG embedding for every row of this family (train+val) --
        # the frozen MACE embedding subset directly when is_mace, otherwise
        # the just-trained SG body's own encode().
        r_sg_family_all = (
            r_soap_all[family_mask_all]
            if is_mace
            else np.asarray(sg_body.encode(r_soap_all[family_mask_all]))
        )
        family_positions = np.flatnonzero(family_mask_all)
        pos_lookup = {row: i for i, row in enumerate(family_positions)}
        train_pos = np.array([pos_lookup[i] for i in np.flatnonzero(family_mask_train)])
        val_pos = np.array([pos_lookup[i] for i in np.flatnonzero(expert_val_mask)])
        local_sg_ids_family = local_ids_full[family_mask_all]

        # 2. Classifier on top of the frozen SG embedding.
        sg_classifier = ClassificationTail(
            input_dim=sg_repr_dim,
            hidden_dim=hs.sg_classifier_hidden_dim,
            n_family_classes=len(local_classes),
            seed=config.seed,
        )
        sg_classifier_history = train_classification_tail(
            sg_classifier,
            r_sg_family_all[train_pos],
            r_sg_family_all[val_pos],
            tail_train_config,
            train_family_ids=local_sg_ids_family[train_pos],
            val_family_ids=local_sg_ids_family[val_pos],
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
            tail_train_config,
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
            r_sg_predicted = (
                r_soap_all[predicted_mask]
                if is_mace
                else np.asarray(sg_body.encode(r_soap_all[predicted_mask]))
            )
            local_probs = np.asarray(
                jax.nn.softmax(sg_classifier.classify_family(r_sg_predicted), axis=-1)
            )
            spacegroup_probs_all[np.ix_(predicted_mask, cols)] = local_probs
        if true_mask.any():
            local_probs_oracle = np.asarray(
                jax.nn.softmax(sg_classifier.classify_family(r_sg_family_all), axis=-1)
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


def _train_hierarchical_visualization(
    config: TailTrainConfig,
    tail_dir: Path,
    run_dir: Path,
    loaded,
) -> None:
    """Train one ``VisualizationTail`` *per family*, attached on top of an
    already-trained hierarchical tail's own per-family spacegroup experts
    (frozen, never retrained here) -- one level deeper than the usual
    body -> tail freezing pattern: here it's expert -> tail. See
    ``dim_red.pipeline.config.HierarchicalVisualizationConfig`` for the
    full rationale/field docs.

    Either the referenced expert's last hidden layer activation
    (``ClassificationTail.family_hidden``, ``input_source: "expert"``, the
    default) or the frozen body's own embedding directly
    (``input_source: "body"``, in parallel with the expert rather than
    chained to it) is the input, computed for *every* point of that family
    (train + val -- the same full subset the expert itself was
    trained/evaluated on). Families with no dedicated expert in the
    referenced hierarchical tail are skipped either way -- see
    ``dim_red.pipeline.config.HierarchicalVisualizationConfig``.
    """
    hv = config.hierarchical_visualization
    source_tail_dir = run_dir / "tails" / hv.hierarchical_output_subdir
    status_path = source_tail_dir / "family_expert_status.yaml"
    if not status_path.exists():
        raise FileNotFoundError(
            f"{status_path} not found -- {source_tail_dir} doesn't look like "
            "a completed tail_kind='hierarchical' tail (see "
            "hierarchical_visualization.hierarchical_output_subdir)"
        )
    with open(status_path) as f:
        status_by_family = {
            entry["family"]: entry for entry in yaml.safe_load(f)["families"]
        }

    labels_all = loaded.embeddings["labels"]
    spacegroups_all = loaded.embeddings["spacegroups"]
    material_ids_all = loaded.embeddings["material_ids"]
    split_all = loaded.embeddings["split"]
    train_idx = split_all == "train"
    val_idx = split_all == "val"

    use_expert = hv.input_source == "expert"
    if use_expert:
        with open(source_tail_dir / "tail_config.yaml") as f:
            source_hierarchical = yaml.safe_load(f)["hierarchical"]
        source_expert_input = source_hierarchical.get("expert_input", "body")
        source_expert_hidden_dim = (
            source_hierarchical.get("expert_head_hidden_dim")
            or source_hierarchical["head_hidden_dim"]
        )
        source_seed = source_hierarchical.get("seed", 42)
        if source_expert_input == "soap":
            logger.info(
                "Recomputing native SOAP features from %s to match the "
                "referenced hierarchical tail's own expert_input='soap'",
                run_dir / "dataset.extxyz",
            )
            r_expert_source = _compute_native_soap_features(
                run_dir, loaded.config, loaded.embeddings
            )
        else:
            r_expert_source = loaded.embeddings["embeddings"]
    else:
        # input_source: "body" -- in parallel with the expert (same
        # relationship the family-level classifier/visualization tail
        # already have), no need to reconstruct any expert or recompute
        # native SOAP at all.
        r_expert_source = loaded.embeddings["embeddings"]
    expert_input_dim = r_expert_source.shape[1]

    family_classes, family_ids = _build_vocab_ids(labels_all.tolist())
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
    balanced_batching = hv.batching.strategy == "balanced"

    plot_fn = plot_reduced_space if hv.viz_dim == 2 else plot_reduced_space_3d
    trained_families = []
    for k, family_name in enumerate(family_classes):
        status = status_by_family.get(family_name)
        if status is None or not status.get("expert", False):
            logger.info(
                "Family %r has no dedicated expert in %s -- skipping its "
                "visualization tail",
                family_name,
                source_tail_dir,
            )
            continue

        expert_dir = source_tail_dir / "experts" / _sanitize_family_dirname(family_name)
        with open(expert_dir / "local_spacegroup_classes.yaml") as f:
            local_classes = yaml.safe_load(f)["local_spacegroup_classes"]

        family_mask_all = family_ids == k
        family_mask_train = train_idx & family_mask_all
        family_mask_val = val_idx & family_mask_all
        if not family_mask_val.any():
            family_mask_val = family_mask_train

        if use_expert:
            expert_tail = load_classification_tail(
                expert_dir / "tail_params.msgpack",
                input_dim=expert_input_dim,
                hidden_dim=source_expert_hidden_dim,
                n_family_classes=len(local_classes),
                seed=source_seed,
            )
            r_family = np.asarray(
                expert_tail.family_hidden(r_expert_source[family_mask_all])
            )
        else:
            # input_source: "body" -- attach directly to the frozen
            # body's own embedding, in parallel with the expert, exactly
            # like the family-level classifier/visualization tail pair.
            r_family = np.asarray(r_expert_source[family_mask_all])
        family_positions = np.flatnonzero(family_mask_all)
        pos_lookup = {row: i for i, row in enumerate(family_positions)}
        train_pos = np.array([pos_lookup[i] for i in np.flatnonzero(family_mask_train)])
        val_pos = np.array([pos_lookup[i] for i in np.flatnonzero(family_mask_val)])

        local_spacegroups_all = spacegroups_all[family_mask_all]
        local_class_to_id = {c: i for i, c in enumerate(local_classes)}
        local_sg_ids = np.array(
            [local_class_to_id[sg] for sg in local_spacegroups_all.tolist()]
        )

        viz_hidden_dim = hv.hidden_dim or [r_family.shape[1]]
        viz_tail = VisualizationTail(
            input_dim=r_family.shape[1],
            hidden_dim=viz_hidden_dim,
            output_dim=hv.viz_dim,
            seed=config.seed,
        )
        logger.info(
            "Training visualization tail for family %r: input_source=%s "
            "input_dim=%d n_local_spacegroups=%d viz_dim=%d tau=%.3f "
            "distance=%s epochs=%d batch_size=%d batching=%s",
            family_name,
            hv.input_source,
            r_family.shape[1],
            len(local_classes),
            hv.viz_dim,
            hv.tau,
            hv.distance,
            config.train.epochs,
            config.train.batch_size,
            hv.batching.strategy,
        )
        history = train_visualization_tail(
            viz_tail,
            r_family[train_pos],
            r_family[val_pos],
            train_config,
            train_spacegroup_ids=local_sg_ids[train_pos],
            val_spacegroup_ids=local_sg_ids[val_pos],
            lambda_family=0.0,
            lambda_spacegroup=1.0,
            lambda_norm=hv.lambda_norm,
            batching_strategy=hv.batching.strategy,
            # Balanced batching stratifies by "family" -- here there is no
            # family dimension left (this is already one family's subset),
            # so the local spacegroup ids are passed AS the "family" axis,
            # with a dummy all-zero "spacegroup" axis (the documented
            # pattern for when the finer sub-stratification isn't needed --
            # see dim_red.supcon.sampling.iter_balanced_batches).
            batching_family_ids=(
                local_sg_ids[train_pos] if balanced_batching else None
            ),
            batching_spacegroup_ids=(
                np.zeros_like(local_sg_ids[train_pos]) if balanced_batching else None
            ),
            batching_P=hv.batching.balanced_params.P,
            batching_K=hv.batching.balanced_params.K,
            batching_S=None,
        )
        actual_epochs = len(history["train_loss"])
        logger.info(
            "Family %r visualization tail: %d/%d epochs, final train_loss=%.4f "
            "val_loss=%.4f",
            family_name,
            actual_epochs,
            config.train.epochs,
            history["train_loss"][-1],
            history["val_loss"][-1],
        )

        family_dir = tail_dir / _sanitize_family_dirname(family_name)
        family_dir.mkdir(parents=True, exist_ok=True)
        _save_loss_history(family_dir / "loss_history.csv", history)
        with open(family_dir / "tail_params.msgpack", "wb") as f:
            f.write(serialization.to_bytes(viz_tail.params))

        z_family = np.asarray(viz_tail.project(r_family))
        family_split = np.where(family_mask_train[family_positions], "train", "val")
        np.savez(
            family_dir / "tail_embeddings.npz",
            embeddings=z_family,
            spacegroups=local_spacegroups_all,
            material_ids=material_ids_all[family_positions],
            split=family_split,
        )
        plot_path = family_dir / "viz_plot_spacegroup.png"
        plot_fn(
            z_family,
            [str(sg) for sg in local_spacegroups_all.tolist()],
            title=f"{family_name} visualization ({hv.input_source} input, {run_dir.name})",
            save_path=str(plot_path),
        )
        logger.info("Saved family %r visualization plot to %s", family_name, plot_path)
        trained_families.append(family_name)

    logger.info(
        "Trained %d/%d per-family visualization tails (skipped families with "
        "no dedicated expert): %s",
        len(trained_families),
        len(family_classes),
        trained_families,
    )


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
            r_expert_all = None
            if config.hierarchical.expert_input == "soap":
                logger.info(
                    "hierarchical.expert_input='soap' -- recomputing native "
                    "SOAP features from %s for stage-2 experts",
                    run_dir / "dataset.extxyz",
                )
                r_expert_all = _compute_native_soap_features(
                    run_dir,
                    loaded.config,
                    loaded.embeddings,
                )
                logger.info(
                    "Recomputed native SOAP features: shape=%s", r_expert_all.shape
                )
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
                r_expert_all=r_expert_all,
            )
            logger.info("Tail artifacts saved to %s", tail_dir)
        elif config.tail_kind == "hierarchical_visualization":
            with open(tail_dir / "tail_config.yaml", "w") as f:
                yaml.safe_dump(tail_train_config_to_dict(config), f, sort_keys=False)
            _train_hierarchical_visualization(config, tail_dir, run_dir, loaded)
            logger.info("Tail artifacts saved to %s", tail_dir)
        elif config.tail_kind == "hierarchical_supcon":
            with open(tail_dir / "tail_config.yaml", "w") as f:
                yaml.safe_dump(tail_train_config_to_dict(config), f, sort_keys=False)
            _train_hierarchical_supcon(config, tail_dir, run_dir, loaded)
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
