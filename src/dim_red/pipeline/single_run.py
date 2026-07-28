"""
Executes one fetch -> SOAP -> VAE training pass from a RunConfig and persists
every artifact needed for later analysis: the config used, the trained model,
per-epoch loss curves (total/recon/KL, train and val), the latent embeddings
of every point in the dataset used for training, and a 2D scatter plot of
those embeddings.
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import yaml
from flax import serialization

from dim_red.analysis.plotting import plot_reduced_space
from dim_red.pipeline.config import RunConfig, run_config_to_dict
from dim_red.pipeline.dataset_cache import get_or_build_dataset
from dim_red.vae.database import VAEDatabase
from dim_red.vae.model import VAE
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
    return f"{timestamp}_hd-{hd}_cs-{cs}"


def _split_indices(n_samples: int, val_ratio: float, seed: int) -> tuple:
    """Reproducible train/val index split (at least one sample per side)."""
    n_val = max(1, int(round(n_samples * val_ratio)))
    n_val = min(n_val, n_samples - 1)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    return indices[n_val:], indices[:n_val]


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
        (latent embeddings of every point in the training dataset),
        ``embeddings_plot.png`` and ``run.log``.
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
        X, labels, material_ids = get_or_build_dataset(
            crystal_systems=config.crystal_systems,
            soap_kwargs=config.soap.as_kwargs(),
            limit_per_system=config.limit_per_system,
            cache_dir=resolved_cache_dir,
            api_key=config.api_key,
        )
        logger.info("Dataset ready: X.shape=%s, %d samples", X.shape, len(labels))

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
            mirror=config.vae.mirror,
            seed=config.seed,
        )

        train_config = TrainConfig(
            epochs=config.train.epochs,
            batch_size=config.train.batch_size,
            learning_rate=config.train.learning_rate,
            beta=config.train.beta,
            seed=config.seed,
            device=config.train.device,
        )
        logger.info(
            "Training VAE: encoder_hidden_dim=%s latent_dim=%d epochs=%d "
            "batch_size=%d beta=%.3f device=%s",
            config.vae.encoder_hidden_dim,
            config.vae.latent_dim,
            config.train.epochs,
            config.train.batch_size,
            config.train.beta,
            config.train.device,
        )

        history = train_vae(model, train_db, val_db, train_config)
        for epoch, (tl, tr, tk, vl, vr, vk) in enumerate(
            zip(
                history["train_loss"],
                history["train_recon"],
                history["train_kl"],
                history["val_loss"],
                history["val_recon"],
                history["val_kl"],
            ),
            start=1,
        ):
            logger.info(
                "epoch %d/%d - train_loss=%.4f (recon=%.4f kl=%.4f) - "
                "val_loss=%.4f (recon=%.4f kl=%.4f)",
                epoch,
                config.train.epochs,
                tl,
                tr,
                tk,
                vl,
                vr,
                vk,
            )

        _save_loss_history(run_dir / "loss_history.csv", history)

        # Apply the trained encoder to every point of the dataset used for
        # training (train + val), not just the held-out validation split.
        mu_all, _ = model.encode(X)
        mu_all = np.asarray(mu_all)
        np.savez(
            run_dir / "embeddings.npz",
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
