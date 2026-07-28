"""
Executes one fetch -> SOAP -> VAE training pass from a RunConfig and persists
every artifact needed for later analysis: the config used, the trained model,
per-epoch loss curves (total/recon/KL, train and val, plus family/spacegroup
cross-entropy when auxiliary heads are active), the latent embeddings of
every point in the dataset used for training (with soft-masked auxiliary
head predictions when active), and a 2D scatter plot of those embeddings.
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from flax import serialization

from dim_red.analysis.plotting import plot_reduced_space
from dim_red.pipeline.config import RunConfig, run_config_to_dict
from dim_red.pipeline.dataset_cache import get_or_build_dataset
from dim_red.vae.database import VAEDatabase
from dim_red.vae.model import VAE, apply_family_mask
from dim_red.vae.training import TrainConfig, train_vae

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


def make_run_name(config: RunConfig) -> str:
    """Build a run directory name encoding the swept parameters, unless the
    config sets an explicit ``name``.
    """
    if config.name:
        return config.name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    hd = _slugify(config.vae.encoder_hidden_dim)
    cs = _slugify(sorted(cs.lower() for cs in config.crystal_systems))
    name = f"{timestamp}_hd-{hd}_cs-{cs}"
    aux = config.aux_heads
    if aux.mode != "none":
        name += f"_aux-{aux.mode}_lf{aux.lambda_family:g}"
        if aux.mode == "family_and_spacegroup":
            name += f"_lsg{aux.lambda_spacegroup:g}"
    return name


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


def _log_epoch(epoch: int, total_epochs: int, history: Dict[str, List[float]]) -> None:
    """Log one epoch's losses, including whichever auxiliary keys are present."""
    i = epoch - 1
    msg = (
        f"epoch {epoch}/{total_epochs} - "
        f"train_loss={history['train_loss'][i]:.4f} "
        f"(recon={history['train_recon'][i]:.4f} kl={history['train_kl'][i]:.4f}"
    )
    if "train_family_ce" in history:
        msg += f" family_ce={history['train_family_ce'][i]:.4f}"
    if "train_spacegroup_ce" in history:
        msg += f" spacegroup_ce={history['train_spacegroup_ce'][i]:.4f}"
    msg += (
        f") - val_loss={history['val_loss'][i]:.4f} "
        f"(recon={history['val_recon'][i]:.4f} kl={history['val_kl'][i]:.4f}"
    )
    if "val_family_ce" in history:
        msg += f" family_ce={history['val_family_ce'][i]:.4f}"
    if "val_spacegroup_ce" in history:
        msg += f" spacegroup_ce={history['val_spacegroup_ce'][i]:.4f}"
    msg += ")"
    logger.info(msg)


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


def run_single(config: RunConfig, cache_dir: Optional[Union[str, Path]] = None) -> Path:
    """Run one fetch -> SOAP -> VAE training pass and persist all artifacts.

    Returns:
        Path to the run directory containing ``config.yaml``,
        ``model_params.msgpack``, ``loss_history.csv``, ``embeddings.npz``
        (latent embeddings of every point in the training dataset, plus
        ``family_probs``/``spacegroup_probs`` and their class vocabularies
        when auxiliary heads are active), ``embeddings_plot.png`` and
        ``run.log``.
    """
    output_dir = Path(config.output_dir)
    run_dir = output_dir / make_run_name(config)
    run_dir.mkdir(parents=True, exist_ok=False)

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
        X, labels, material_ids, spacegroups = get_or_build_dataset(
            crystal_systems=config.crystal_systems,
            soap_kwargs=config.soap.as_kwargs(),
            limit_per_system=config.limit_per_system,
            cache_dir=resolved_cache_dir,
            api_key=config.api_key,
        )
        logger.info("Dataset ready: X.shape=%s, %d samples", X.shape, len(labels))

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
            logger.info(
                "Auxiliary heads active (mode=%s): %d family classes: %s",
                aux_mode,
                len(family_classes),
                family_classes,
            )
        if use_spacegroup:
            n_unknown = sum(1 for sg in spacegroups if sg < 0)
            if n_unknown:
                logger.warning(
                    "%d/%d structures have no MP spacegroup data; treating -1 as its "
                    "own spacegroup class",
                    n_unknown,
                    len(spacegroups),
                )
            spacegroup_classes, spacegroup_ids = _build_vocab_ids(spacegroups)
            family_spacegroup_mask = _build_family_spacegroup_mask(
                family_ids, spacegroup_ids, len(family_classes), len(spacegroup_classes)
            )
            logger.info("%d spacegroup classes observed", len(spacegroup_classes))

        train_idx, val_idx = _split_indices(
            X.shape[0], config.train.val_ratio, config.seed
        )
        train_db = VAEDatabase.from_array(X[train_idx])
        val_db = VAEDatabase.from_array(X[val_idx])
        split = np.full(X.shape[0], "train", dtype="<U5")
        split[val_idx] = "val"
        logger.info(
            "Train/val split: %d train / %d val (val_ratio=%.2f, seed=%d)",
            train_db.data.shape[0],
            val_db.data.shape[0],
            config.train.val_ratio,
            config.seed,
        )

        model = VAE(
            input_dim=X.shape[1],
            encoder_hidden_dim=config.vae.encoder_hidden_dim,
            decoder_hidden_dim=config.vae.decoder_hidden_dim,
            latent_dim=config.vae.latent_dim,
            n_family_classes=len(family_classes) if use_family else None,
            n_spacegroup_classes=len(spacegroup_classes) if use_spacegroup else None,
            head_hidden_dim=config.aux_heads.head_hidden_dim,
            mirror=config.vae.mirror,
            seed=config.seed,
        )

        train_config = TrainConfig(
            epochs=config.train.epochs,
            batch_size=config.train.batch_size,
            learning_rate=config.train.learning_rate,
            beta=config.train.beta,
            lambda_family=config.aux_heads.lambda_family,
            lambda_spacegroup=config.aux_heads.lambda_spacegroup,
            seed=config.seed,
            device=config.train.device,
        )
        logger.info(
            "Training VAE: encoder_hidden_dim=%s latent_dim=%d epochs=%d "
            "batch_size=%d beta=%.3f aux_heads=%s device=%s",
            config.vae.encoder_hidden_dim,
            config.vae.latent_dim,
            config.train.epochs,
            config.train.batch_size,
            config.train.beta,
            aux_mode,
            config.train.device,
        )

        history = train_vae(
            model,
            train_db,
            val_db,
            train_config,
            train_family_ids=family_ids[train_idx] if use_family else None,
            val_family_ids=family_ids[val_idx] if use_family else None,
            train_spacegroup_ids=spacegroup_ids[train_idx] if use_spacegroup else None,
            val_spacegroup_ids=spacegroup_ids[val_idx] if use_spacegroup else None,
            family_spacegroup_mask=family_spacegroup_mask if use_spacegroup else None,
        )
        for epoch in range(1, config.train.epochs + 1):
            _log_epoch(epoch, config.train.epochs, history)

        _save_loss_history(run_dir / "loss_history.csv", history)

        # Apply the trained encoder to every point of the dataset used for
        # training (train + val), not just the held-out validation split.
        mu_all, _ = model.encode(X)
        mu_all = np.asarray(mu_all)
        embeddings_payload = dict(
            embeddings=mu_all,
            labels=np.array(labels),
            material_ids=np.array(material_ids),
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
                    f"VAE latent space (hidden={config.vae.encoder_hidden_dim}, "
                    f"crystal_systems={config.crystal_systems})"
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

        logger.info("Run complete: artifacts saved to %s", run_dir)
    finally:
        logger.removeHandler(file_handler)
        file_handler.close()

    return run_dir
