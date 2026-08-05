"""
Unit tests for VAE training loop utilities.
"""

import pytest

pytest.importorskip("jax")

import jax
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

    # No auxiliary keys when no family/spacegroup ids are given.
    assert "train_family_ce" not in history
    assert "train_spacegroup_ce" not in history


def _make_split_db(n=40, n_features=5, val_ratio=0.25, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features)).astype(np.float32)
    db = VAEDatabase.from_array(X)
    return db.train_val_split(val_ratio=val_ratio, seed=seed)


def _split_indices_like_train_val_split(n, val_ratio, seed):
    """Reproduce VAEDatabase.train_val_split's index split to align external
    label arrays (family/spacegroup ids) with train_db/val_db rows.
    """
    n_val = max(1, int(round(n * val_ratio)))
    n_val = min(n_val, n - 1)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    return indices[n_val:], indices[:n_val]


def test_train_vae_family_only_aux_loss():
    n = 40
    train_db, val_db = _make_split_db(n=n)
    train_idx, val_idx = _split_indices_like_train_val_split(n, 0.25, 0)

    rng = np.random.default_rng(1)
    n_family = 3
    family_ids = rng.integers(0, n_family, size=n).astype(np.int32)

    model = VAE(
        input_dim=5,
        encoder_hidden_dim=[8],
        decoder_hidden_dim=None,
        latent_dim=2,
        seed=0,
        n_family_classes=n_family,
        head_hidden_dim=4,
    )
    config = TrainConfig(
        epochs=2, batch_size=8, beta=1.0, lambda_family=0.7, seed=0, device="cpu"
    )
    history = train_vae(
        model,
        train_db,
        val_db,
        config,
        train_family_ids=family_ids[train_idx],
        val_family_ids=family_ids[val_idx],
    )

    assert "train_family_ce" in history and "val_family_ce" in history
    assert "train_spacegroup_ce" not in history
    assert len(history["train_family_ce"]) == 2

    for total, recon, kl, family_ce in zip(
        history["train_loss"],
        history["train_recon"],
        history["train_kl"],
        history["train_family_ce"],
    ):
        assert total == pytest.approx(
            recon + config.beta * kl + config.lambda_family * family_ce, abs=1e-5
        )


def test_train_vae_family_and_spacegroup_aux_loss():
    n = 40
    train_db, val_db = _make_split_db(n=n)
    train_idx, val_idx = _split_indices_like_train_val_split(n, 0.25, 0)

    rng = np.random.default_rng(1)
    n_family = 3
    n_spacegroup = 6
    family_ids = rng.integers(0, n_family, size=n).astype(np.int32)
    # Spacegroup deterministic given family, so the mask is meaningful.
    spacegroup_ids = (family_ids * 2 + rng.integers(0, 2, size=n)).astype(np.int32)
    mask = np.zeros((n_family, n_spacegroup), dtype=np.float32)
    mask[family_ids, spacegroup_ids] = 1.0

    model = VAE(
        input_dim=5,
        encoder_hidden_dim=[8],
        decoder_hidden_dim=None,
        latent_dim=2,
        seed=0,
        n_family_classes=n_family,
        n_spacegroup_classes=n_spacegroup,
        head_hidden_dim=4,
    )
    config = TrainConfig(
        epochs=2,
        batch_size=8,
        beta=1.0,
        lambda_family=1.0,
        lambda_spacegroup=0.5,
        seed=0,
        device="cpu",
    )
    history = train_vae(
        model,
        train_db,
        val_db,
        config,
        train_family_ids=family_ids[train_idx],
        val_family_ids=family_ids[val_idx],
        train_spacegroup_ids=spacegroup_ids[train_idx],
        val_spacegroup_ids=spacegroup_ids[val_idx],
        family_spacegroup_mask=mask,
    )

    for key in (
        "train_family_ce",
        "val_family_ce",
        "train_spacegroup_ce",
        "val_spacegroup_ce",
    ):
        assert key in history
        assert len(history[key]) == 2

    for total, recon, kl, family_ce, spacegroup_ce in zip(
        history["train_loss"],
        history["train_recon"],
        history["train_kl"],
        history["train_family_ce"],
        history["train_spacegroup_ce"],
    ):
        assert total == pytest.approx(
            recon
            + config.beta * kl
            + config.lambda_family * family_ce
            + config.lambda_spacegroup * spacegroup_ce,
            abs=1e-5,
        )


def test_train_vae_spacegroup_ids_require_family_and_mask():
    n = 40
    train_db, val_db = _make_split_db(n=n)
    train_idx, val_idx = _split_indices_like_train_val_split(n, 0.25, 0)
    rng = np.random.default_rng(1)
    spacegroup_ids = rng.integers(0, 6, size=n).astype(np.int32)

    model = VAE(
        input_dim=5,
        encoder_hidden_dim=[8],
        decoder_hidden_dim=None,
        latent_dim=2,
        seed=0,
        n_family_classes=3,
        n_spacegroup_classes=6,
        head_hidden_dim=4,
    )
    config = TrainConfig(epochs=1, batch_size=8, seed=0, device="cpu")

    with pytest.raises(ValueError, match="require train_family_ids"):
        train_vae(
            model,
            train_db,
            val_db,
            config,
            train_spacegroup_ids=spacegroup_ids[train_idx],
            val_spacegroup_ids=spacegroup_ids[val_idx],
        )


def test_train_vae_family_ids_must_be_given_with_val_ids():
    n = 40
    train_db, val_db = _make_split_db(n=n)
    train_idx, _ = _split_indices_like_train_val_split(n, 0.25, 0)
    rng = np.random.default_rng(1)
    family_ids = rng.integers(0, 3, size=n).astype(np.int32)

    model = VAE(
        input_dim=5,
        encoder_hidden_dim=[8],
        decoder_hidden_dim=None,
        latent_dim=2,
        seed=0,
        n_family_classes=3,
        head_hidden_dim=4,
    )
    config = TrainConfig(epochs=1, batch_size=8, seed=0, device="cpu")

    with pytest.raises(ValueError, match="must be given together"):
        train_vae(
            model, train_db, val_db, config, train_family_ids=family_ids[train_idx]
        )


def test_train_vae_early_stopping_disabled_runs_full_epochs():
    train_db, val_db = _make_split_db(n=40)
    model = VAE(
        input_dim=5,
        encoder_hidden_dim=[8],
        decoder_hidden_dim=None,
        latent_dim=2,
        seed=0,
    )
    config = TrainConfig(epochs=4, batch_size=8, seed=0, device="cpu")
    history = train_vae(model, train_db, val_db, config)
    assert len(history["train_loss"]) == 4


def test_train_vae_early_stopping_stops_before_configured_epochs():
    train_db, val_db = _make_split_db(n=40)
    model = VAE(
        input_dim=5,
        encoder_hidden_dim=[8],
        decoder_hidden_dim=None,
        latent_dim=2,
        seed=0,
    )
    config = TrainConfig(
        epochs=30,
        batch_size=8,
        seed=0,
        device="cpu",
        early_stopping=True,
        early_stopping_patience=2,
    )
    history = train_vae(model, train_db, val_db, config)
    n_epochs_run = len(history["train_loss"])
    assert n_epochs_run < config.epochs
    assert len(history["val_loss"]) == n_epochs_run
    # The best val_loss must have been reached strictly before the last
    # early_stopping_patience epochs -- that's exactly why training stopped.
    best_epoch_idx = int(np.argmin(history["val_loss"]))
    assert best_epoch_idx <= n_epochs_run - 1 - config.early_stopping_patience


def test_train_vae_early_stopping_min_delta_requires_larger_improvement():
    """A large min_delta makes small improvements not count, so training
    should stop earlier (or equal) than with min_delta=0 given the same seed.
    """
    train_db, val_db = _make_split_db(n=40)

    model_loose = VAE(
        input_dim=5,
        encoder_hidden_dim=[8],
        decoder_hidden_dim=None,
        latent_dim=2,
        seed=0,
    )
    config_loose = TrainConfig(
        epochs=30,
        batch_size=8,
        seed=0,
        device="cpu",
        early_stopping=True,
        early_stopping_patience=2,
        early_stopping_min_delta=0.0,
    )
    history_loose = train_vae(model_loose, train_db, val_db, config_loose)

    model_strict = VAE(
        input_dim=5,
        encoder_hidden_dim=[8],
        decoder_hidden_dim=None,
        latent_dim=2,
        seed=0,
    )
    config_strict = TrainConfig(
        epochs=30,
        batch_size=8,
        seed=0,
        device="cpu",
        early_stopping=True,
        early_stopping_patience=2,
        early_stopping_min_delta=10.0,
    )
    history_strict = train_vae(model_strict, train_db, val_db, config_strict)

    assert len(history_strict["train_loss"]) <= len(history_loose["train_loss"])


def test_train_vae_early_stopping_restore_best_changes_final_params():
    train_db, val_db = _make_split_db(n=40)

    model_restore = VAE(
        input_dim=5,
        encoder_hidden_dim=[8],
        decoder_hidden_dim=None,
        latent_dim=2,
        seed=0,
    )
    config_restore = TrainConfig(
        epochs=30,
        batch_size=8,
        seed=0,
        device="cpu",
        early_stopping=True,
        early_stopping_patience=2,
        early_stopping_restore_best=True,
    )
    history_restore = train_vae(model_restore, train_db, val_db, config_restore)

    model_no_restore = VAE(
        input_dim=5,
        encoder_hidden_dim=[8],
        decoder_hidden_dim=None,
        latent_dim=2,
        seed=0,
    )
    config_no_restore = TrainConfig(
        epochs=30,
        batch_size=8,
        seed=0,
        device="cpu",
        early_stopping=True,
        early_stopping_patience=2,
        early_stopping_restore_best=False,
    )
    history_no_restore = train_vae(
        model_no_restore, train_db, val_db, config_no_restore
    )

    # Same trajectory either way -- restore_best only changes which
    # snapshot ends up in model.params, not the training run itself.
    assert history_restore["val_loss"] == history_no_restore["val_loss"]
    n_epochs_run = len(history_restore["val_loss"])
    assert n_epochs_run < config_restore.epochs  # confirm it actually stopped early

    best_epoch_idx = int(np.argmin(history_restore["val_loss"]))
    assert best_epoch_idx != n_epochs_run - 1  # the best epoch wasn't the last one

    leaves_restore = jax.tree_util.tree_leaves(model_restore.params)
    leaves_no_restore = jax.tree_util.tree_leaves(model_no_restore.params)
    assert any(
        not np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(leaves_restore, leaves_no_restore)
    )


def test_train_vae_early_stopping_rejects_invalid_patience():
    with pytest.raises(ValueError, match="early_stopping_patience"):
        train_vae(
            VAE(
                input_dim=5,
                encoder_hidden_dim=[8],
                decoder_hidden_dim=None,
                latent_dim=2,
                seed=0,
            ),
            *_make_split_db(n=40),
            TrainConfig(
                epochs=1,
                batch_size=8,
                seed=0,
                device="cpu",
                early_stopping=True,
                early_stopping_patience=0,
            ),
        )


def test_train_vae_early_stopping_rejects_negative_min_delta():
    with pytest.raises(ValueError, match="early_stopping_min_delta"):
        train_vae(
            VAE(
                input_dim=5,
                encoder_hidden_dim=[8],
                decoder_hidden_dim=None,
                latent_dim=2,
                seed=0,
            ),
            *_make_split_db(n=40),
            TrainConfig(
                epochs=1,
                batch_size=8,
                seed=0,
                device="cpu",
                early_stopping=True,
                early_stopping_min_delta=-1.0,
            ),
        )
