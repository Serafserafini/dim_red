"""
Executes one fetch -> SOAP -> model training pass from a RunConfig and
persists every artifact needed for later analysis: the config used, the
trained model, per-epoch loss curves (total/recon, plus KL for a VAE and
family/spacegroup cross-entropy when auxiliary heads are active), the exact
structures used for training (``dataset.extxyz``), the latent embeddings of
every point in the dataset used for training (with soft-masked auxiliary
head predictions when active), and a 2D scatter plot of those embeddings.
``config.model_kind`` selects between a VAE (``dim_red.vae``), a
deterministic Autoencoder (``dim_red.autoencoder``), an encoder-only
Supervised Contrastive body (``dim_red.supcon``, no reconstruction/KL/
classifier heads at all -- its family/spacegroup labels drive a contrastive
loss, gated by ``config.supcon.mode`` instead of ``config.aux_heads.mode``,
computed on a jointly-trained ``dim_red.supcon.tails.ProjectionTail``'s
output rather than the body's own representation -- see
``dim_red.supcon.training``), or a graph-convolutional encoder-only body
(``dim_red.cgcnn``, no decoder/KL/contrastive loss at all -- trained
directly against family/spacegroup labels via cross-entropy, single-phase,
like ``vae``/``autoencoder``'s ``aux_heads`` pattern applied to a
graph-convolutional body instead of a flat-feature encoder; it reads
crystal structures directly, via ``dim_red.cgcnn.graph``, rather than SOAP
descriptors). ``vae``/``autoencoder``/``supcon`` share the same encoder
architecture (``config.vae``); ``cgcnn`` reads its graph-construction +
architecture hyperparameters from ``config.graph`` instead (only
``config.vae.latent_dim`` is still read). ``supcon`` reads its own loss
settings from ``config.supcon`` and ignores ``config.aux_heads``, the
reverse of what ``vae``/``autoencoder``/``cgcnn`` do. A completed
``supcon``/``cgcnn`` run's frozen body can then have a classification or
visualization tail trained on top of it separately -- see
``dim_red.pipeline.tail_training``.
"""

from __future__ import annotations

import csv
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from unicodedata import name

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from flax import serialization

from dim_red.analysis.plotting import plot_reduced_space
from dim_red.autoencoder.model import Autoencoder
from dim_red.autoencoder.model import apply_family_mask as ae_apply_family_mask
from dim_red.autoencoder.training import TrainConfig as AETrainConfig
from dim_red.autoencoder.training import train_autoencoder
from dim_red.cgcnn.database import GraphDatabase
from dim_red.cgcnn.model import CGCNNEncoder
from dim_red.cgcnn.model import apply_family_mask as cgcnn_apply_family_mask
from dim_red.cgcnn.training import TrainConfig as CGCNNTrainConfig
from dim_red.cgcnn.training import train_cgcnn
from dim_red.pipeline.config import RunConfig, TailTrainConfig, run_config_to_dict
from dim_red.pipeline.dataset_cache import (
    build_dataset_for_run,
    build_graph_dataset_for_run,
)
from dim_red.pipeline.tail_training import train_tail
from dim_red.supcon.model import SupConEncoder
from dim_red.supcon.tails import ProjectionTail
from dim_red.supcon.training import TrainConfig as SupConTrainConfig
from dim_red.supcon.training import train_supcon
from dim_red.vae.database import VAEDatabase
from dim_red.vae.model import VAE
from dim_red.vae.model import apply_family_mask as vae_apply_family_mask
from dim_red.vae.training import TrainConfig as VAETrainConfig
from dim_red.vae.training import train_vae

logger = logging.getLogger("dim_red.pipeline")
# `dim_red.vae.training` imports `learned_optimization`, which pulls in absl
# and attaches its own handler to the root logger as an import side effect.
# That makes a caller's `logging.basicConfig(level=INFO)` a no-op (basicConfig
# only configures the root logger if it has no handlers yet), which silently
# drops every INFO record -- including the ones written to each run's
# `run.log`. Set the level explicitly on our own logger so run logging works
# regardless of what the caller (or absl) did to the root logger.
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
    (hidden dims, the data source, aux-head/SupCon lambdas, and the model
    kind when it isn't the default ``"vae"``) -- no timestamp -- so sibling
    runs of the same sweep are identifiable by what they swept, not by when
    they ran. Tagging non-default ``model_kind`` values means a sweep
    varying it (e.g. ``grid: {"model": ["vae", "autoencoder"]}``) still gets
    distinctly named runs rather than colliding.
    """
    if config.name:
        return config.name
    hd = _slugify(config.vae.encoder_hidden_dim)
    name = f"hd-{hd}_{_data_source_slug(config)}"
    if config.model_kind != "vae":
        name = f"model-{config.model_kind}_{name}"
    aux = config.aux_heads
    if aux.mode != "none":
        name += f"_aux-{aux.mode}_lf{aux.lambda_family:g}"
        if aux.mode == "family_and_spacegroup":
            name += f"_lsg{aux.lambda_spacegroup:g}"
    if config.model_kind == "supcon":
        sc = config.supcon
        name += f"_supcon-{sc.mode}_tau{sc.tau:g}"
        if sc.mode != "spacegroup_only":
            name += f"_lf{sc.lambda_family:g}"
        if sc.mode != "family_only":
            name += f"_lsg{sc.lambda_spacegroup:g}"
    return name


def _make_unique_run_dir(output_dir: Path, name: str) -> Path:
    """Create and return ``output_dir / name``, deduplicating with a
    ``-<n>`` suffix if that directory already exists (e.g. re-running the
    same unnamed config into the same ``output_dir``) so runs never silently
    overwrite one another now that names carry no timestamp.
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


def _split_indices(n_samples: int, val_ratio: float, seed: int) -> tuple:
    """Reproducible train/val index split (at least one sample per side)."""
    n_val = max(1, int(round(n_samples * val_ratio)))
    n_val = min(n_val, n_samples - 1)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    return indices[n_val:], indices[:n_val]


def _build_vocab_ids(values: List) -> Tuple[List, np.ndarray]:
    """Map arbitrary hashable values to a sorted vocabulary and integer ids.

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
    under a family in this dataset, 0.0 otherwise. Derived directly from the
    data rather than hardcoded crystallographic spacegroup ranges.
    """
    mask = np.zeros((n_family, n_spacegroup), dtype=np.float32)
    mask[family_ids, spacegroup_ids] = 1.0
    return mask


_EPOCH_COMPONENT_LABELS = (
    ("recon", "recon"),
    ("kl", "kl"),
    ("family_ce", "family_ce"),
    ("spacegroup_ce", "spacegroup_ce"),
    ("family_supcon", "family_supcon"),
    ("spacegroup_supcon", "spacegroup_supcon"),
    ("norm_penalty", "norm_penalty"),
)


def _log_epoch(epoch: int, total_epochs: int, history: Dict[str, List[float]]) -> None:
    """Log one epoch's losses, including whichever component keys are present.

    A VAE's history has ``recon``/``kl`` (plus ``family_ce``/``spacegroup_ce``
    when aux heads are active); a plain Autoencoder's has ``recon`` only (plus
    the same aux-head keys); a SupCon encoder's has neither ``recon`` nor
    ``kl`` at all, just ``family_supcon``/``spacegroup_supcon``. Every
    component is therefore included conditionally, so this one function
    serves all three ``model_kind`` histories without needing to know which
    produced it.
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


def _save_loss_history(path: Path, history: Dict[str, List[float]]) -> None:
    fieldnames = ["epoch"] + list(history.keys())
    n_epochs = len(next(iter(history.values())))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(n_epochs):
            row = {"epoch": epoch + 1}
            row.update({k: v[epoch] for k, v in history.items()})
            writer.writerow(row)


def run_single(
    config: RunConfig,
    cache_dir: Optional[Union[str, Path]] = None,
    save_features: bool = True,
) -> Path:
    """Run one fetch -> SOAP -> VAE training pass and persist all artifacts.

    Args:
        config: The run's configuration.
        cache_dir: Dataset cache directory (default: ``<output_dir>/_dataset_cache``).
        save_features: Whether to include the raw standardized SOAP
            ``features`` matrix in this run's ``embeddings.npz`` (ignored for
            ``model_kind == "cgcnn"``, which never has one -- see below).
            Defaults to ``True`` for a standalone run. ``X`` is identical
            across every run of a sweep that shares the same dataset cache
            key (same crystal systems/pyxtal config + SOAP settings +
            augmentation) -- since ``dim_red.pipeline.compare.compute_embedding_baselines``
            only ever reads ``features`` from one run per sweep, and
            ``dim_red.pipeline.inference.load_trained_run``'s fast path
            (``feature_mean``/``feature_std`` present) doesn't read it at
            all -- saving a full copy into *every* run's ``embeddings.npz``
            is pure duplication once a sweep has more than a couple of runs;
            X alone was measured at multiple GB per run on a real
            ~35000-structure pyxtal dataset. ``dim_red.pipeline.sweep.run_sweep``
            passes ``False`` here for every run after the first one sharing a
            given dataset cache key.

    Returns:
        Path to the run directory containing ``config.yaml``,
        ``model_params.msgpack``, ``loss_history.csv``, ``dataset.extxyz``
        (the exact structures used for training, same order as
        ``embeddings.npz``'s arrays), ``embeddings.npz`` (latent embeddings
        of every point in the training dataset; for ``model_kind in
        ("vae", "autoencoder", "supcon")`` also, when ``save_features`` is
        True, the raw standardized SOAP ``features`` fed to the model --
        ``feature_mean``/``feature_std`` (the standardization stats they
        were derived from) are saved unconditionally, regardless of
        ``save_features``, since ``dim_red.pipeline.inference.load_trained_run``
        needs those for every run, not just one per sweep -- from
        ``dim_red.pipeline.dataset_cache.build_dataset_for_run``, so
        ``dim_red.pipeline.inference.load_trained_run`` can standardize new
        structures the same way without ever recomputing SOAP on this run's
        own training set -- ``features`` omitted entirely for
        ``model_kind == "cgcnn"``, which has no natural flat feature vector
        or standardization step at all (Gaussian-expanded bond features are
        already bounded to ``[0, 1]`` by construction) -- the true
        ``spacegroups`` per point, plus ``family_probs``/``spacegroup_probs``
        and their class vocabularies when auxiliary heads are active --
        never the case for ``model_kind == "supcon"``, which has no
        classifier heads at all), ``embeddings_plot.png`` and ``run.log``.
    """
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
        is_cgcnn = config.model_kind == "cgcnn"
        if is_cgcnn:
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
        else:
            (
                X,
                labels,
                material_ids,
                spacegroups,
                structures_path,
                feature_mean,
                feature_std,
            ) = build_dataset_for_run(config, cache_dir=resolved_cache_dir)
            n_samples = X.shape[0]
            logger.info("Dataset ready: X.shape=%s, %d samples", X.shape, len(labels))

        dataset_path = run_dir / "dataset.extxyz"
        shutil.copyfile(structures_path, dataset_path)
        logger.info("Saved training dataset structures to %s", dataset_path)

        is_supcon = config.model_kind == "supcon"
        if is_supcon:
            aux_mode = config.supcon.mode
            use_family = aux_mode != "spacegroup_only"
            use_spacegroup = aux_mode != "family_only"
        else:
            aux_mode = config.aux_heads.mode
            use_family = aux_mode != "none"
            use_spacegroup = aux_mode == "family_and_spacegroup"

        # Balanced batching (SupCon only) always groups by family first, and
        # always by spacegroup within family too (every spacegroup present
        # by default, or a capped subset via balanced_params.S) --
        # independent of whether those labels also drive a SupCon *loss*
        # term (use_family/use_spacegroup above). E.g. mode="spacegroup_only"
        # + batching.strategy="balanced" still needs family_ids built here,
        # purely for batch construction, even though the family loss term
        # itself stays inactive.
        balanced_batching = is_supcon and config.batching.strategy == "balanced"
        need_family_ids = use_family or balanced_batching
        need_spacegroup_ids = use_spacegroup or balanced_batching

        family_classes: List[str] = []
        spacegroup_classes: List[int] = []
        family_ids = None
        spacegroup_ids = None
        family_spacegroup_mask = None
        if need_family_ids:
            family_classes, family_ids = _build_vocab_ids(labels)
            logger.info("%d family classes: %s", len(family_classes), family_classes)
            if use_family:
                logger.info(
                    "%s active (mode=%s)",
                    "SupCon" if is_supcon else "Auxiliary heads",
                    aux_mode,
                )
            if balanced_batching:
                logger.info(
                    "Balanced batching active: training batches grouped by family%s",
                    " and spacegroup" if need_spacegroup_ids else "",
                )
        if need_spacegroup_ids:
            n_unknown = sum(1 for sg in spacegroups if sg < 0)
            if n_unknown:
                logger.warning(
                    "%d/%d structures have no MP spacegroup data; treating -1 as its "
                    "own spacegroup class",
                    n_unknown,
                    len(spacegroups),
                )
            spacegroup_classes, spacegroup_ids = _build_vocab_ids(spacegroups)
            # SupCon's "spacegroup_only" mode has use_spacegroup=True with
            # use_family=False (no family <-> spacegroup masking involved at
            # all for this model, unlike the vae/autoencoder aux heads, whose
            # use_spacegroup can only be True together with use_family) --
            # skip building the mask in that case, since it's family_ids-
            # dependent and would go unused anyway.
            if use_family:
                family_spacegroup_mask = _build_family_spacegroup_mask(
                    family_ids,
                    spacegroup_ids,
                    len(family_classes),
                    len(spacegroup_classes),
                )
            logger.info("%d spacegroup classes observed", len(spacegroup_classes))

        train_idx, val_idx = _split_indices(
            n_samples, config.train.val_ratio, config.seed
        )
        if is_cgcnn:
            graph_db = GraphDatabase.from_arrays(
                local_species_idx, nbr_idx, nbr_fea, nbr_mask, atom_mask
            )
            train_db = graph_db[train_idx]
            val_db = graph_db[val_idx]
            logger.info(
                "Train/val split: %d train / %d val (val_ratio=%.2f, seed=%d)",
                train_db.n_samples,
                val_db.n_samples,
                config.train.val_ratio,
                config.seed,
            )
        else:
            train_db = VAEDatabase.from_array(X[train_idx])
            val_db = VAEDatabase.from_array(X[val_idx])
            logger.info(
                "Train/val split: %d train / %d val (val_ratio=%.2f, seed=%d)",
                train_db.data.shape[0],
                val_db.data.shape[0],
                config.train.val_ratio,
                config.seed,
            )
        split = np.full(n_samples, "train", dtype="<U5")
        split[val_idx] = "val"

        is_vae = config.model_kind == "vae"

        if is_supcon:
            # No decoder, no classifier heads at all -- the encoder (body)
            # is trained jointly with a projection tail against family/
            # spacegroup labels via a contrastive loss computed on the
            # tail's output, not the body's own representation (Khosla et
            # al. 2020). Its constructor takes none of the aux-head/decoder
            # arguments vae/autoencoder need.
            model = SupConEncoder(
                input_dim=X.shape[1],
                encoder_hidden_dim=config.vae.encoder_hidden_dim,
                latent_dim=config.vae.latent_dim,
                seed=config.seed,
            )
            projection_tail = ProjectionTail(
                input_dim=config.vae.latent_dim,
                hidden_dim=(
                    config.supcon.projection_hidden_dim or [config.vae.latent_dim]
                ),
                projection_dim=config.supcon.projection_dim,
                seed=config.seed,
            )
            # Unused: the family/spacegroup probability-inference block below
            # is guarded by `not is_supcon`, which never touches this branch.
            apply_family_mask = None
        elif is_cgcnn:
            apply_family_mask = cgcnn_apply_family_mask
            model = CGCNNEncoder(
                atom_fea_len=config.graph.atom_fea_len,
                n_conv=config.graph.n_conv,
                h_fea_len=config.graph.h_fea_len,
                n_h=config.graph.n_h,
                latent_dim=config.vae.latent_dim,
                n_gaussian=config.graph.n_gaussian,
                max_species=config.graph.max_species,
                n_family_classes=len(family_classes) if use_family else None,
                n_spacegroup_classes=(
                    len(spacegroup_classes) if use_spacegroup else None
                ),
                head_hidden_dim=config.aux_heads.head_hidden_dim,
                seed=config.seed,
            )
        else:
            apply_family_mask = (
                vae_apply_family_mask if is_vae else ae_apply_family_mask
            )
            model_cls = VAE if is_vae else Autoencoder
            model = model_cls(
                input_dim=X.shape[1],
                encoder_hidden_dim=config.vae.encoder_hidden_dim,
                decoder_hidden_dim=config.vae.decoder_hidden_dim,
                latent_dim=config.vae.latent_dim,
                n_family_classes=len(family_classes) if use_family else None,
                n_spacegroup_classes=(
                    len(spacegroup_classes) if use_spacegroup else None
                ),
                head_hidden_dim=config.aux_heads.head_hidden_dim,
                mirror=config.vae.mirror,
                seed=config.seed,
            )

        # Shared across every model kind's TrainConfig -- same field
        # names on VAETrainConfig/AETrainConfig/SupConTrainConfig/
        # CGCNNTrainConfig, see dim_red.pipeline.config.EarlyStoppingConfig.
        early_stopping_kwargs = dict(
            early_stopping=config.train.early_stopping.enabled,
            early_stopping_patience=config.train.early_stopping.patience,
            early_stopping_min_delta=config.train.early_stopping.min_delta,
            early_stopping_restore_best=config.train.early_stopping.restore_best_weights,
        )

        if is_vae:
            train_config = VAETrainConfig(
                epochs=config.train.epochs,
                batch_size=config.train.batch_size,
                learning_rate=config.train.learning_rate,
                optimizer=config.train.optimizer,
                beta=config.train.beta,
                lambda_family=config.aux_heads.lambda_family,
                lambda_spacegroup=config.aux_heads.lambda_spacegroup,
                seed=config.seed,
                device=config.train.device,
                **early_stopping_kwargs,
            )
            logger.info(
                "Training VAE: encoder_hidden_dim=%s latent_dim=%d epochs=%d "
                "batch_size=%d beta=%.3f aux_heads=%s device=%s optimizer=%s "
                "early_stopping=%s",
                config.vae.encoder_hidden_dim,
                config.vae.latent_dim,
                config.train.epochs,
                config.train.batch_size,
                config.train.beta,
                aux_mode,
                config.train.device,
                config.train.optimizer,
                config.train.early_stopping.enabled,
            )
        elif is_supcon:
            train_config = SupConTrainConfig(
                epochs=config.train.epochs,
                batch_size=config.train.batch_size,
                learning_rate=config.train.learning_rate,
                optimizer=config.train.optimizer,
                tau=config.supcon.tau,
                distance=config.supcon.distance,
                seed=config.seed,
                device=config.train.device,
                **early_stopping_kwargs,
            )
            logger.info(
                "Training SupCon body+projection tail: encoder_hidden_dim=%s "
                "latent_dim=%d projection_dim=%d projection_hidden_dim=%s "
                "epochs=%d batch_size=%d tau=%.3f distance=%s lambda_norm=%.3f "
                "mode=%s device=%s optimizer=%s batching=%s%s early_stopping=%s",
                config.vae.encoder_hidden_dim,
                config.vae.latent_dim,
                config.supcon.projection_dim,
                config.supcon.projection_hidden_dim or [config.vae.latent_dim],
                config.train.epochs,
                config.train.batch_size,
                config.supcon.tau,
                config.supcon.distance,
                config.supcon.lambda_norm,
                aux_mode,
                config.train.device,
                config.train.optimizer,
                config.batching.strategy,
                (
                    f" (P={config.batching.balanced_params.P} "
                    f"K={config.batching.balanced_params.K} "
                    f"S={config.batching.balanced_params.S})"
                    if balanced_batching
                    else ""
                ),
                config.train.early_stopping.enabled,
            )
        elif is_cgcnn:
            train_config = CGCNNTrainConfig(
                epochs=config.train.epochs,
                batch_size=config.train.batch_size,
                learning_rate=config.train.learning_rate,
                optimizer=config.train.optimizer,
                lambda_family=config.aux_heads.lambda_family,
                lambda_spacegroup=config.aux_heads.lambda_spacegroup,
                seed=config.seed,
                device=config.train.device,
                **early_stopping_kwargs,
            )
            logger.info(
                "Training CGCNN: atom_fea_len=%d n_conv=%d h_fea_len=%d n_h=%d "
                "latent_dim=%d radius=%.1f max_num_nbr=%d n_gaussian=%d epochs=%d "
                "batch_size=%d aux_heads=%s device=%s optimizer=%s early_stopping=%s",
                config.graph.atom_fea_len,
                config.graph.n_conv,
                config.graph.h_fea_len,
                config.graph.n_h,
                config.vae.latent_dim,
                config.graph.radius,
                config.graph.max_num_nbr,
                config.graph.n_gaussian,
                config.train.epochs,
                config.train.batch_size,
                aux_mode,
                config.train.device,
                config.train.optimizer,
                config.train.early_stopping.enabled,
            )
        else:
            train_config = AETrainConfig(
                epochs=config.train.epochs,
                batch_size=config.train.batch_size,
                learning_rate=config.train.learning_rate,
                optimizer=config.train.optimizer,
                lambda_family=config.aux_heads.lambda_family,
                lambda_spacegroup=config.aux_heads.lambda_spacegroup,
                seed=config.seed,
                device=config.train.device,
                **early_stopping_kwargs,
            )
            logger.info(
                "Training Autoencoder: encoder_hidden_dim=%s latent_dim=%d epochs=%d "
                "batch_size=%d aux_heads=%s device=%s optimizer=%s early_stopping=%s",
                config.vae.encoder_hidden_dim,
                config.vae.latent_dim,
                config.train.epochs,
                config.train.batch_size,
                aux_mode,
                config.train.device,
                config.train.optimizer,
                config.train.early_stopping.enabled,
            )

        if is_supcon:
            history = train_supcon(
                model,
                projection_tail,
                train_db,
                val_db,
                train_config,
                train_family_ids=family_ids[train_idx] if use_family else None,
                val_family_ids=family_ids[val_idx] if use_family else None,
                train_spacegroup_ids=(
                    spacegroup_ids[train_idx] if use_spacegroup else None
                ),
                val_spacegroup_ids=spacegroup_ids[val_idx] if use_spacegroup else None,
                lambda_family=config.supcon.lambda_family,
                lambda_spacegroup=config.supcon.lambda_spacegroup,
                lambda_norm=config.supcon.lambda_norm,
                batching_strategy=config.batching.strategy,
                batching_family_ids=(
                    family_ids[train_idx] if balanced_batching else None
                ),
                batching_spacegroup_ids=(
                    spacegroup_ids[train_idx] if balanced_batching else None
                ),
                batching_P=config.batching.balanced_params.P,
                batching_K=config.batching.balanced_params.K,
                batching_S=config.batching.balanced_params.S,
            )
        elif is_cgcnn:
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
                family_spacegroup_mask=(
                    family_spacegroup_mask if use_spacegroup else None
                ),
            )
        else:
            train_fn = train_vae if is_vae else train_autoencoder
            history = train_fn(
                model,
                train_db,
                val_db,
                train_config,
                train_family_ids=family_ids[train_idx] if use_family else None,
                val_family_ids=family_ids[val_idx] if use_family else None,
                train_spacegroup_ids=(
                    spacegroup_ids[train_idx] if use_spacegroup else None
                ),
                val_spacegroup_ids=spacegroup_ids[val_idx] if use_spacegroup else None,
                family_spacegroup_mask=(
                    family_spacegroup_mask if use_spacegroup else None
                ),
            )
        # Actual epoch count, not config.train.epochs: early stopping (see
        # dim_red.pipeline.config.EarlyStoppingConfig) can make history
        # shorter than the configured epochs, and indexing _log_epoch past
        # the end of a shortened history would raise.
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
        # A VAE's encode returns (mu, logvar); an Autoencoder's/SupConEncoder's/
        # CGCNNEncoder's returns just z, since encoding is deterministic (no
        # posterior to describe).
        if is_cgcnn:
            mu_all = model.encode(
                (local_species_idx, nbr_idx, nbr_fea, nbr_mask, atom_mask)
            )
        elif is_vae:
            mu_all = model.encode(X)[0]
        else:
            mu_all = model.encode(X)
        mu_all = np.asarray(mu_all)
        if is_cgcnn:
            # No natural flat feature vector or standardization step exists
            # for graph features (Gaussian-expanded bond distances are
            # already bounded to [0, 1] by construction) -- features/
            # feature_mean/feature_std are omitted entirely for this
            # model_kind. A consequence: dim_red.pipeline.compare's
            # classical-baseline (PCA/UMAP-on-features) comparisons have
            # nothing to fit against for cgcnn runs.
            embeddings_payload = dict(
                embeddings=mu_all,
                labels=np.array(labels),
                material_ids=np.array(material_ids),
                spacegroups=np.array(spacegroups, dtype=np.int64),
                split=split,
            )
        else:
            embeddings_payload = dict(
                embeddings=mu_all,
                labels=np.array(labels),
                material_ids=np.array(material_ids),
                spacegroups=np.array(spacegroups, dtype=np.int64),
                split=split,
                # Per-feature standardization stats these SOAP features were
                # derived from (dim_red.pipeline.dataset_cache._compute_soap_and_standardize),
                # so dim_red.pipeline.inference.load_trained_run can standardize
                # new structures the same way without ever recomputing SOAP on
                # this run's own training set. Saved unconditionally (unlike
                # "features" below) since every run needs its own copy for that.
                feature_mean=feature_mean,
                feature_std=feature_std,
            )
            if save_features:
                # Raw standardized SOAP features (the model's actual input),
                # so downstream comparison tooling (see
                # dim_red.pipeline.compare.compute_embedding_baselines) can
                # fit classical baselines (PCA, UMAP) on the exact same data
                # without needing to re-fetch/re-run SOAP. X is identical
                # across every run in a sweep sharing this run's dataset
                # cache key (see this function's save_features docs), so
                # dim_red.pipeline.sweep.run_sweep only requests this for the
                # first such run rather than duplicating a multi-GB array
                # into every run's embeddings.npz.
                embeddings_payload["features"] = X
        logger.info(
            "Encoded %d points into %d-dim latent space",
            mu_all.shape[0],
            mu_all.shape[1],
        )

        # Pure inference pass over the whole dataset (train + val), using the
        # trained heads with SOFT masking (predicted family), since -- unlike
        # during training -- no true family label is used here: this mirrors
        # genuine downstream inference where the true family is unknown.
        # SupCon has no classifier heads at all (the labels are only ever
        # used inside the contrastive loss during training, never at
        # inference) so this whole block is skipped for it -- embeddings.npz
        # keeps only the base fields (embeddings/features/labels/
        # material_ids/spacegroups/split) for that model_kind. CGCNN, unlike
        # SupCon, does have classifier heads (its only training objective),
        # so this block runs for it -- family_probs/spacegroup_probs are
        # saved same as vae/autoencoder, just alongside a smaller base
        # payload (no features/feature_mean/feature_std, see above).
        if use_family and not is_supcon:
            family_logits_all = model.classify_family(mu_all)
            family_probs_all = np.asarray(jax.nn.softmax(family_logits_all, axis=-1))
            embeddings_payload["family_probs"] = family_probs_all
            embeddings_payload["family_classes"] = np.array(family_classes)
            logger.info("Saved family_probs for %d points", family_probs_all.shape[0])

            if use_spacegroup:
                spacegroup_logits_all = model.classify_spacegroup(mu_all)
                masked_logits_all = apply_family_mask(
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

        if config.vae.latent_dim >= 2:
            plot_path = run_dir / "embeddings_plot.png"
            plot_reduced_space(
                mu_all,
                labels,
                title=(
                    f"HD={'->'.join(str(dim) for dim in config.vae.encoder_hidden_dim)}, "
                    f"{_data_source_slug(config)}"
                ),
                save_path=str(plot_path),
                xlabel="Latent Dimension 1",
                ylabel="Latent Dimension 2",
            )
            logger.info("Saved 2D latent-space plot to %s", plot_path)
        else:
            logger.warning(
                "latent_dim=%d < 2: skipping 2D embeddings plot", config.vae.latent_dim
            )

        with open(run_dir / "model_params.msgpack", "wb") as f:
            f.write(serialization.to_bytes(model.params))

        if is_supcon:
            # Discarded in the original paper once phase-1 training is
            # done -- kept here purely for reproducibility/debugging the
            # exact loss surface that produced the body. Nothing downstream
            # (dim_red.pipeline.inference/tail_training) ever reloads it.
            with open(run_dir / "projection_params.msgpack", "wb") as f:
                f.write(serialization.to_bytes(projection_tail.params))
            logger.info(
                "Saved projection tail params to %s",
                run_dir / "projection_params.msgpack",
            )

        # Auto-train tails (dim_red.pipeline.config.RunConfig.tails): the
        # exact same dim_red.pipeline.tail_training.train_tail entry point
        # dimred-train-tail invokes manually, just triggered automatically
        # right after phase-1 training finishes. train_tail only re-reads
        # config.yaml/embeddings.npz from run_dir (already written above) via
        # load_run_embeddings -- it never touches dataset.extxyz and never
        # recomputes SOAP/graphs, so this is the same cost a manual
        # dimred-train-tail call would incur, not extra cost from automating
        # it. Its own run.log FileHandler is
        # added to this same "dim_red.pipeline" logger while ours is still
        # attached, so this run's run.log ends up containing a full trace of
        # any auto-triggered tail training too.
        if (is_supcon or is_cgcnn) and config.tails is not None:
            if config.tails.classification is not None:
                logger.info("Auto-training classification tail on this run")
                tail_dir = train_tail(
                    TailTrainConfig(
                        run_dir=str(run_dir),
                        tail_kind="classification",
                        classification=config.tails.classification,
                        train=config.tails.train,
                        seed=config.seed,
                    )
                )
                logger.info("Auto-trained classification tail saved to %s", tail_dir)
            if config.tails.visualization is not None:
                logger.info("Auto-training visualization tail on this run")
                tail_dir = train_tail(
                    TailTrainConfig(
                        run_dir=str(run_dir),
                        tail_kind="visualization",
                        visualization=config.tails.visualization,
                        train=config.tails.train,
                        seed=config.seed,
                    )
                )
                logger.info("Auto-trained visualization tail saved to %s", tail_dir)

        logger.info("Run complete: artifacts saved to %s", run_dir)
    finally:
        logger.removeHandler(file_handler)
        file_handler.close()

    return run_dir
