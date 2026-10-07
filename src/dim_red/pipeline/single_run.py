"""
Executes one structures -> graphs -> CGCNN training pass from a RunConfig
and persists every artifact needed for later analysis: the config used, the
trained model, per-epoch loss curves, the exact structures used for training
(``dataset.extxyz``), the latent embeddings of every point in the dataset
used for training (with soft-masked classifier-head predictions), and a 2D
scatter plot of those embeddings.

``run_single`` only trains ``config.model_kind == "cgcnn"``: a
graph-convolutional encoder (``dim_red.cgcnn``) trained directly against
family/spacegroup labels via cross-entropy in a single phase
(``config.aux_heads``); it reads crystal structures directly, via
``dim_red.cgcnn.graph``. It reads its graph-construction and architecture
hyperparameters from ``config.graph``, and only ``config.encoder.latent_dim``
from the shared ``encoder`` block. Every other model kind is trained by
``dim_red.pipeline.full_stack.FullStack``. A completed run's frozen body can
then have a classification or visualization tail trained on top of it
separately -- see ``dim_red.pipeline.tail_training``.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from flax import serialization

from dim_red.analysis.plotting import plot_reduced_space
from dim_red.cgcnn.database import GraphDatabase
from dim_red.cgcnn.model import CGCNNEncoder
from dim_red.cgcnn.model import apply_family_mask as cgcnn_apply_family_mask
from dim_red.cgcnn.training import TrainConfig as CGCNNTrainConfig
from dim_red.cgcnn.training import train_cgcnn
from dim_red.pipeline._common import (
    _build_vocab_ids,
    _make_unique_run_dir,
    _save_loss_history,
    _split_indices_grouped,
)
from dim_red.pipeline.config import RunConfig, run_config_to_dict
from dim_red.pipeline.dataset_cache import build_graph_dataset_for_run

logger = logging.getLogger("dim_red.pipeline")
# Set the level explicitly on our own logger so run logging (including each
# run's `run.log`) works regardless of how the caller configured the root
# logger.
logger.setLevel(logging.INFO)


def _slugify(values) -> str:
    return "-".join(str(v) for v in values)


def _data_source_slug(config: RunConfig) -> str:
    """The dataset-defining segment of a run name: ``cs-<crystal systems>``
    for ``data_source == "fetch"``, or ``pyxtal-<scope>_nsp<n_species>`` for
    ``data_source == "pyxtal"``.
    """
    if config.data_source == "pyxtal":
        pc = config.pyxtal
        if pc.spacegroups:
            scope = _slugify(sorted(pc.spacegroups))
        elif pc.families:
            scope = _slugify(sorted(f.lower()[:3] for f in pc.families))
        else:
            scope = "all"
        return f"pyxtal-{scope}_nsp{pc.n_species}"
    cs = _slugify(sorted(cs.lower() for cs in config.fetch.crystal_systems))
    return f"cs-{cs}"


def make_run_name(config: RunConfig) -> str:
    """Build a run directory name encoding the swept parameters, unless the
    config sets an explicit ``name``.

    The name carries only the hyperparameters that distinguish this run
    (the model kind, hidden dims, the data source, aux-head lambdas)
    -- no timestamp -- so sibling runs of the same sweep are identifiable by
    what they swept, not by when they ran. The model kind is always part of
    the name, so a sweep varying it still gets distinctly named runs rather
    than colliding.
    """
    if config.name:
        return config.name
    hd = _slugify(config.encoder.encoder_hidden_dim)
    name = f"model-{config.model_kind}_hd-{hd}_{_data_source_slug(config)}"
    aux = config.aux_heads
    if aux.mode != "none":
        name += f"_aux-{aux.mode}_lf{aux.lambda_family:g}"
        if aux.mode == "family_and_spacegroup":
            name += f"_lsg{aux.lambda_spacegroup:g}"
    return name


def _build_family_spacegroup_mask(
    family_ids: np.ndarray, spacegroup_ids: np.ndarray, n_family: int, n_spacegroup: int
) -> np.ndarray:
    """Empirical co-occurrence mask: 1.0 where a spacegroup was observed
    under a family in this dataset, 0.0 otherwise. Derived directly from the
    data rather than hardcoded crystallographic spacegroup ranges.
    """
    mask = np.zeros((n_family, n_spacegroup), dtype=np.float32)
    mask[family_ids, spacegroup_ids] = 1.0
    return mask


_EPOCH_COMPONENT_LABELS = (
    ("family_ce", "family_ce"),
    ("spacegroup_ce", "spacegroup_ce"),
)


def _log_epoch(epoch: int, total_epochs: int, history: Dict[str, List[float]]) -> None:
    """Log one epoch's losses, including whichever component keys are present.

    A CGCNN's history has ``family_ce`` and, with spacegroup heads active,
    ``spacegroup_ce``; each component is included only when present.
    """
    i = epoch - 1

    def _components(prefix: str) -> str:
        parts = [
            f"{label}={history[f'{prefix}_{key}'][i]:.4f}"
            for key, label in _EPOCH_COMPONENT_LABELS
            if f"{prefix}_{key}" in history
        ]
        return " ".join(parts)

    logger.info(
        "epoch %d/%d - train_loss=%.4f (%s) - val_loss=%.4f (%s)",
        epoch,
        total_epochs,
        history["train_loss"][i],
        _components("train"),
        history["val_loss"][i],
        _components("val"),
    )


def run_single(
    config: RunConfig,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Run one structures -> graphs -> CGCNN training pass and persist all artifacts.

    Args:
        config: The run's configuration; ``config.model_kind`` must be
            ``"cgcnn"`` (supcon/supcon_mace runs use
            ``dim_red.pipeline.full_stack.FullStack``).
        cache_dir: Dataset cache directory (default: ``<output_dir>/_dataset_cache``).

    Returns:
        Path to the run directory containing ``config.yaml``,
        ``model_params.msgpack``, ``loss_history.csv``, ``dataset.extxyz``
        (the exact structures used for training, same order as
        ``embeddings.npz``'s arrays), ``embeddings.npz`` (latent embeddings
        of every point in the training dataset, the true ``spacegroups`` per
        point, plus ``family_probs``/``spacegroup_probs`` and their class
        vocabularies when auxiliary heads are active; no ``features``/
        ``feature_mean``/``feature_std``, since graph features have no
        natural flat feature vector or standardization step --
        Gaussian-expanded bond features are already bounded to ``[0, 1]``
        by construction), ``embeddings_plot.png`` and ``run.log``.

    Raises:
        ValueError: If ``config.model_kind`` is not ``"cgcnn"``.
    """
    if config.model_kind != "cgcnn":
        raise ValueError(
            f"run_single only trains model_kind 'cgcnn', got {config.model_kind!r}. "
            "supcon/supcon_mace runs use dim_red.pipeline.full_stack.FullStack "
            "(dimred-run with a model_kind: config)."
        )
    output_dir = Path(config.output_dir)
    run_dir = _make_unique_run_dir(output_dir, make_run_name(config))

    file_handler = logging.FileHandler(run_dir / "run.log")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logger.addHandler(file_handler)
    try:
        logger.info("Starting run in %s", run_dir)
        with open(run_dir / "config.yaml", "w") as f:
            yaml.safe_dump(run_config_to_dict(config), f, sort_keys=False)

        resolved_cache_dir = (
            Path(cache_dir) if cache_dir else output_dir / "_dataset_cache"
        )
        (
            graph_arrays,
            labels,
            material_ids,
            spacegroups,
            structures_path,
        ) = build_graph_dataset_for_run(config, cache_dir=resolved_cache_dir)
        local_species_idx, nbr_idx, nbr_fea, nbr_mask, atom_mask = graph_arrays
        n_samples = local_species_idx.shape[0]
        logger.info(
            "Dataset ready: %d samples, max_atoms=%d, max_num_nbr=%d",
            n_samples,
            local_species_idx.shape[1],
            nbr_idx.shape[2],
        )

        dataset_path = run_dir / "dataset.extxyz"
        shutil.copyfile(structures_path, dataset_path)
        logger.info("Saved training dataset structures to %s", dataset_path)

        aux_mode = config.aux_heads.mode
        use_family = aux_mode != "none"
        use_spacegroup = aux_mode == "family_and_spacegroup"

        family_classes: List[str] = []
        spacegroup_classes: List[int] = []
        family_ids = None
        spacegroup_ids = None
        family_spacegroup_mask = None
        if use_family:
            family_classes, family_ids = _build_vocab_ids(labels)
            logger.info("%d family classes: %s", len(family_classes), family_classes)
            logger.info("Auxiliary heads active (mode=%s)", aux_mode)
        if use_spacegroup:
            n_unknown = sum(1 for sg in spacegroups if sg < 0)
            if n_unknown:
                logger.warning(
                    "%d/%d structures have no spacegroup data; treating -1 as its "
                    "own spacegroup class",
                    n_unknown,
                    len(spacegroups),
                )
            spacegroup_classes, spacegroup_ids = _build_vocab_ids(spacegroups)
            # use_spacegroup implies use_family for the aux heads, so the
            # family <-> spacegroup co-occurrence mask can always be built.
            family_spacegroup_mask = _build_family_spacegroup_mask(
                family_ids,
                spacegroup_ids,
                len(family_classes),
                len(spacegroup_classes),
            )
            logger.info("%d spacegroup classes observed", len(spacegroup_classes))

        train_idx, val_idx = _split_indices_grouped(
            material_ids, config.train.val_ratio, config.seed
        )
        graph_db = GraphDatabase.from_arrays(
            local_species_idx, nbr_idx, nbr_fea, nbr_mask, atom_mask
        )
        train_db = graph_db[train_idx]
        val_db = graph_db[val_idx]
        logger.info(
            "Train/val split (grouped by material_id): %d train / %d val "
            "(val_ratio=%.2f, seed=%d)",
            train_db.n_samples,
            val_db.n_samples,
            config.train.val_ratio,
            config.seed,
        )
        split = np.full(n_samples, "train", dtype="<U5")
        split[val_idx] = "val"

        model = CGCNNEncoder(
            atom_fea_len=config.graph.atom_fea_len,
            n_conv=config.graph.n_conv,
            h_fea_len=config.graph.h_fea_len,
            n_h=config.graph.n_h,
            latent_dim=config.encoder.latent_dim,
            n_gaussian=config.graph.n_gaussian,
            max_species=config.graph.max_species,
            n_family_classes=len(family_classes) if use_family else None,
            n_spacegroup_classes=(len(spacegroup_classes) if use_spacegroup else None),
            head_hidden_dim=config.aux_heads.head_hidden_dim,
            seed=config.seed,
        )

        # See dim_red.pipeline.config.EarlyStoppingConfig.
        train_config = CGCNNTrainConfig(
            epochs=config.train.epochs,
            batch_size=config.train.batch_size,
            learning_rate=config.train.learning_rate,
            lambda_family=config.aux_heads.lambda_family,
            lambda_spacegroup=config.aux_heads.lambda_spacegroup,
            seed=config.seed,
            device=config.train.device,
            early_stopping=config.train.early_stopping.enabled,
            early_stopping_patience=config.train.early_stopping.patience,
            early_stopping_min_delta=config.train.early_stopping.min_delta,
            early_stopping_restore_best=config.train.early_stopping.restore_best_weights,
        )
        logger.info(
            "Training CGCNN: atom_fea_len=%d n_conv=%d h_fea_len=%d n_h=%d "
            "latent_dim=%d radius=%.1f max_num_nbr=%d n_gaussian=%d epochs=%d "
            "batch_size=%d aux_heads=%s device=%s early_stopping=%s",
            config.graph.atom_fea_len,
            config.graph.n_conv,
            config.graph.h_fea_len,
            config.graph.n_h,
            config.encoder.latent_dim,
            config.graph.radius,
            config.graph.max_num_nbr,
            config.graph.n_gaussian,
            config.train.epochs,
            config.train.batch_size,
            aux_mode,
            config.train.device,
            config.train.early_stopping.enabled,
        )

        history = train_cgcnn(
            model,
            train_db,
            val_db,
            train_config,
            train_family_ids=family_ids[train_idx],
            val_family_ids=family_ids[val_idx],
            train_spacegroup_ids=(
                spacegroup_ids[train_idx] if use_spacegroup else None
            ),
            val_spacegroup_ids=spacegroup_ids[val_idx] if use_spacegroup else None,
            family_spacegroup_mask=(family_spacegroup_mask if use_spacegroup else None),
        )
        if history:
            # Actual epoch count, not config.train.epochs: early stopping
            # (see dim_red.pipeline.config.EarlyStoppingConfig) can make
            # history shorter than the configured epochs, and indexing
            # _log_epoch past the end of a shortened history would raise.
            actual_epochs = len(history["train_loss"])
            for epoch in range(1, actual_epochs + 1):
                _log_epoch(epoch, actual_epochs, history)
            if actual_epochs < config.train.epochs:
                logger.info(
                    "Early stopping: training stopped after %d/%d epochs "
                    "(patience=%d, min_delta=%g)",
                    actual_epochs,
                    config.train.epochs,
                    config.train.early_stopping.patience,
                    config.train.early_stopping.min_delta,
                )
            _save_loss_history(run_dir / "loss_history.csv", history)

        # Apply the trained encoder to every point of the dataset used for
        # training (train + val), not just the held-out validation split.
        # Encoding is deterministic, so encode() returns just z.
        mu_all = model.encode(
            (local_species_idx, nbr_idx, nbr_fea, nbr_mask, atom_mask)
        )
        mu_all = np.asarray(mu_all)
        # No natural flat feature vector or standardization step exists for
        # graph features (Gaussian-expanded bond distances are already
        # bounded to [0, 1] by construction) -- features/feature_mean/
        # feature_std are omitted entirely. A consequence:
        # dim_red.pipeline.compare's classical-baseline (PCA/UMAP-on-features)
        # comparisons have nothing to fit against for cgcnn runs.
        embeddings_payload = dict(
            embeddings=mu_all,
            labels=np.array(labels),
            material_ids=np.array(material_ids),
            spacegroups=np.array(spacegroups, dtype=np.int64),
            split=split,
        )
        logger.info(
            "Encoded %d points into %d-dim latent space",
            mu_all.shape[0],
            mu_all.shape[1],
        )

        # Pure inference pass over the whole dataset (train + val), using the
        # trained heads with SOFT masking (predicted family), since -- unlike
        # during training -- no true family label is used here: this mirrors
        # genuine downstream inference where the true family is unknown.
        if use_family:
            family_logits_all = model.classify_family(mu_all)
            family_probs_all = np.asarray(jax.nn.softmax(family_logits_all, axis=-1))
            embeddings_payload["family_probs"] = family_probs_all
            embeddings_payload["family_classes"] = np.array(family_classes)
            logger.info("Saved family_probs for %d points", family_probs_all.shape[0])

            if use_spacegroup:
                spacegroup_logits_all = model.classify_spacegroup(mu_all)
                masked_logits_all = cgcnn_apply_family_mask(
                    spacegroup_logits_all,
                    family_probs_all,
                    jnp.asarray(family_spacegroup_mask),
                )
                spacegroup_probs_all = np.asarray(
                    jax.nn.softmax(masked_logits_all, axis=-1)
                )
                embeddings_payload["spacegroup_probs"] = spacegroup_probs_all
                embeddings_payload["spacegroup_classes"] = np.array(spacegroup_classes)
                logger.info(
                    "Saved spacegroup_probs for %d points",
                    spacegroup_probs_all.shape[0],
                )

        np.savez(run_dir / "embeddings.npz", **embeddings_payload)

        # mu_all.shape[1] always equals config.encoder.latent_dim.
        if mu_all.shape[1] >= 2:
            plot_path = run_dir / "embeddings_plot.png"
            plot_reduced_space(
                mu_all,
                labels,
                title=(
                    f"HD={'->'.join(str(dim) for dim in config.encoder.encoder_hidden_dim)}, "
                    f"{_data_source_slug(config)}"
                ),
                save_path=str(plot_path),
                xlabel="Latent Dimension 1",
                ylabel="Latent Dimension 2",
            )
            logger.info("Saved 2D latent-space plot to %s", plot_path)
        else:
            logger.warning(
                "latent_dim=%d < 2: skipping 2D embeddings plot", mu_all.shape[1]
            )

        with open(run_dir / "model_params.msgpack", "wb") as f:
            f.write(serialization.to_bytes(model.params))

        logger.info("Run complete: artifacts saved to %s", run_dir)
    finally:
        logger.removeHandler(file_handler)
        file_handler.close()

    return run_dir
