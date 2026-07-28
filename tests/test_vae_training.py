"""
Unit tests for VAE training loop utilities.
"""

import pytest

pytest.importorskip("jax")

import numpy as np

from dim_red.vae.database import VAEDatabase
from dim_red.vae.model import VAE
from dim_red.vae.training import TrainConfig, train_vae


def test_train_vae_returns_history():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5)).astype(np.float32)
    db = VAEDatabase.from_array(X)
    train_db, val_db = db.train_val_split(val_ratio=0.25, seed=0)
    model = VAE(
        input_dim=5,
        encoder_hidden_dim=[8],
        decoder_hidden_dim=None,
        latent_dim=2,
        seed=0,
    )

    config = TrainConfig(
        epochs=2, batch_size=8, learning_rate=1e-3, beta=1.0, seed=0, device="cpu"
    )
    history = train_vae(model, train_db, val_db, config)

    for key in (
        "train_loss",
        "train_recon",
        "train_kl",
        "val_loss",
        "val_recon",
        "val_kl",
    ):
        assert len(history[key]) == 2
        assert all(loss >= 0.0 for loss in history[key])

    # total == recon + beta * kl for every epoch, given beta=1.0 in config.
    for total, recon, kl in zip(
        history["train_loss"], history["train_recon"], history["train_kl"]
    ):
        assert total == pytest.approx(recon + config.beta * kl, abs=1e-5)
    for total, recon, kl in zip(
        history["val_loss"], history["val_recon"], history["val_kl"]
    ):
        assert total == pytest.approx(recon + config.beta * kl, abs=1e-5)
