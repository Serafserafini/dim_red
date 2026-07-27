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
    model = VAE(input_dim=5, hidden_dim=8, latent_dim=2, seed=0)

    config = TrainConfig(epochs=2, batch_size=8, learning_rate=1e-3, beta=1.0, seed=0, device="cpu")
    history = train_vae(model, train_db, val_db, config)

    assert len(history["train_loss"]) == 2
    assert len(history["val_loss"]) == 2
    assert all(loss >= 0.0 for loss in history["train_loss"])
    assert all(loss >= 0.0 for loss in history["val_loss"])
