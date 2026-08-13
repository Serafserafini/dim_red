"""
Unit tests for the SupCon (Supervised Contrastive) loss and training loop.
"""

import logging
import math

import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp
import numpy as np

from dim_red.supcon.model import SupConEncoder
from dim_red.supcon.tails import ProjectionTail
from dim_red.supcon.training import TrainConfig, norm_penalty, supcon_loss, train_supcon
from dim_red.vae.database import VAEDatabase


def test_supcon_loss_matches_manual_computation():
    # Two identical points (label 0, mutual positives, squared distance 0)
    # plus one point at squared distance 2 from both (label 1, a singleton
    # -- no positive, excluded).
    z = jnp.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    labels = jnp.array([0, 0, 1])
    tau = 1.0

    loss = supcon_loss(z, labels, tau)

    # By hand: sim_ij = -||z_i - z_j||^2 / tau, so sim(0,1) = -0 = 0 (the
    # positive) and sim(0,2) = -2 (the only negative in A(0) = {1, 2}).
    # per_anchor(0) = -[sim(0,1) - logsumexp(sim(0,1), sim(0,2))]
    #               = logsumexp(0, -2) - 0 = log(1 + e^-2), identical for
    # anchor 1 by symmetry; anchor 2 (label 1) has no positive and is
    # excluded from the mean entirely (not zeroed-and-diluted over 3 anchors).
    expected = math.log(1.0 + math.exp(-2.0))
    assert float(loss) == pytest.approx(expected, abs=1e-5)


def test_supcon_loss_excludes_singleton_anchor_without_nan():
    z = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.9, 0.1]])
    labels = jnp.array([0, 1, 0, 2])  # label 2 (row 3) is a singleton

    loss = supcon_loss(z, labels, tau=0.1)
    assert math.isfinite(float(loss))

    grad = jax.grad(lambda z: supcon_loss(z, labels, tau=0.1))(z)
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_supcon_loss_all_singleton_labels_is_zero_with_zero_grad():
    key = jax.random.PRNGKey(1)
    z = jax.random.normal(key, (5, 4))
    labels = jnp.arange(5)  # every label unique -> no positives anywhere

    loss = supcon_loss(z, labels, tau=0.2)
    assert float(loss) == 0.0

    grad = jax.grad(lambda z: supcon_loss(z, labels, tau=0.2))(z)
    np.testing.assert_array_equal(np.asarray(grad), np.zeros_like(np.asarray(grad)))


def test_supcon_loss_batch_size_one_returns_zero():
    loss = supcon_loss(jnp.array([[1.0, 0.0]]), jnp.array([0]), tau=0.1)
    assert float(loss) == 0.0


def test_supcon_loss_cosine_matches_manual_computation():
    # Two orthogonal unit vectors (label 0, mutual positives) plus one point
    # anti-parallel to the first (label 1, a singleton -- no positive).
    z = jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    labels = jnp.array([0, 0, 1])
    tau = 1.0

    loss = supcon_loss(z, labels, tau, distance="cosine")

    # cosine sim: (0,1)=0, (0,2)=-1, (1,2)=0. Anchor 0: A(0)={1,2},
    # sim/tau={0,-1} -> per_anchor(0) = logsumexp(0,-1) - sim(0,1) =
    # log(1+e^-1). Anchor 1: A(1)={0,2}, sim/tau={0,0} -> per_anchor(1) =
    # logsumexp(0,0) - sim(1,0) = log(2). Anchor 2 (label 1) is a singleton,
    # excluded. Mean over anchors 0 and 1.
    expected = (math.log(1.0 + math.exp(-1.0)) + math.log(2.0)) / 2.0
    assert float(loss) == pytest.approx(expected, abs=1e-5)


def test_supcon_loss_cosine_scale_invariant():
    z = jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.5]])
    labels = jnp.array([0, 0, 1])
    tau = 0.5

    loss_unit = supcon_loss(z, labels, tau, distance="cosine")
    loss_scaled = supcon_loss(z * 100.0, labels, tau, distance="cosine")
    assert float(loss_unit) == pytest.approx(float(loss_scaled), abs=1e-5)


def test_supcon_loss_rejects_invalid_distance():
    z = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    labels = jnp.array([0, 1])
    with pytest.raises(ValueError, match="distance"):
        supcon_loss(z, labels, tau=0.1, distance="manhattan")


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


def _make_projection_tail(latent_dim=3, projection_dim=4, seed=0):
    """A small ProjectionTail matching the SupConEncoder fixtures below --
    projection_dim deliberately != latent_dim so a regression computing
    supcon_loss on the body's raw output instead of the tail's would fail
    with a shape mismatch rather than silently passing.
    """
    return ProjectionTail(
        input_dim=latent_dim, hidden_dim=[4], projection_dim=projection_dim, seed=seed
    )


def test_train_supcon_family_and_spacegroup_returns_history():
    n = 40
    train_db, val_db = _make_split_db(n=n)
    train_idx, val_idx = _split_indices_like_train_val_split(n, 0.25, 0)

    rng = np.random.default_rng(1)
    family_ids = rng.integers(0, 3, size=n).astype(np.int32)
    spacegroup_ids = rng.integers(0, 6, size=n).astype(np.int32)

    model = SupConEncoder(input_dim=5, encoder_hidden_dim=[8], latent_dim=3, seed=0)
    config = TrainConfig(epochs=2, batch_size=8, tau=0.1, seed=0, device="cpu")
    history = train_supcon(
        model,
        _make_projection_tail(),
        train_db,
        val_db,
        config,
        train_family_ids=family_ids[train_idx],
        val_family_ids=family_ids[val_idx],
        train_spacegroup_ids=spacegroup_ids[train_idx],
        val_spacegroup_ids=spacegroup_ids[val_idx],
        lambda_family=1.0,
        lambda_spacegroup=0.5,
    )

    for key in (
        "train_loss",
        "val_loss",
        "train_family_supcon",
        "val_family_supcon",
        "train_spacegroup_supcon",
        "val_spacegroup_supcon",
    ):
        assert key in history
        assert len(history[key]) == 2
        assert all(math.isfinite(v) for v in history[key])
        assert all(v >= 0.0 for v in history[key])


def test_train_supcon_family_only_mode_has_no_spacegroup_keys():
    n = 40
    train_db, val_db = _make_split_db(n=n)
    train_idx, val_idx = _split_indices_like_train_val_split(n, 0.25, 0)
    rng = np.random.default_rng(1)
    family_ids = rng.integers(0, 3, size=n).astype(np.int32)

    model = SupConEncoder(input_dim=5, encoder_hidden_dim=[8], latent_dim=3, seed=0)
    config = TrainConfig(epochs=1, batch_size=8, tau=0.1, seed=0, device="cpu")
    history = train_supcon(
        model,
        _make_projection_tail(),
        train_db,
        val_db,
        config,
        train_family_ids=family_ids[train_idx],
        val_family_ids=family_ids[val_idx],
        lambda_family=1.0,
    )

    assert "train_family_supcon" in history
    assert "train_spacegroup_supcon" not in history
    # No spacegroup term active -> total loss equals the family term exactly.
    for total, family in zip(history["train_loss"], history["train_family_supcon"]):
        assert total == pytest.approx(family, abs=1e-6)


def test_train_supcon_spacegroup_only_mode_has_no_family_keys():
    n = 40
    train_db, val_db = _make_split_db(n=n)
    train_idx, val_idx = _split_indices_like_train_val_split(n, 0.25, 0)
    rng = np.random.default_rng(1)
    spacegroup_ids = rng.integers(0, 6, size=n).astype(np.int32)

    model = SupConEncoder(input_dim=5, encoder_hidden_dim=[8], latent_dim=3, seed=0)
    config = TrainConfig(epochs=1, batch_size=8, tau=0.1, seed=0, device="cpu")
    history = train_supcon(
        model,
        _make_projection_tail(),
        train_db,
        val_db,
        config,
        train_spacegroup_ids=spacegroup_ids[train_idx],
        val_spacegroup_ids=spacegroup_ids[val_idx],
        lambda_spacegroup=1.0,
    )

    assert "train_spacegroup_supcon" in history
    assert "train_family_supcon" not in history


def test_train_supcon_requires_at_least_one_label_type():
    train_db, val_db = _make_split_db(n=40)
    model = SupConEncoder(input_dim=5, encoder_hidden_dim=[8], latent_dim=3, seed=0)
    config = TrainConfig(epochs=1, batch_size=8, seed=0, device="cpu")

    with pytest.raises(ValueError, match="nothing to contrast on"):
        train_supcon(model, _make_projection_tail(), train_db, val_db, config)


def test_train_supcon_family_ids_must_be_given_with_val_ids():
    n = 40
    train_db, val_db = _make_split_db(n=n)
    train_idx, _ = _split_indices_like_train_val_split(n, 0.25, 0)
    rng = np.random.default_rng(1)
    family_ids = rng.integers(0, 3, size=n).astype(np.int32)

    model = SupConEncoder(input_dim=5, encoder_hidden_dim=[8], latent_dim=3, seed=0)
    config = TrainConfig(epochs=1, batch_size=8, seed=0, device="cpu")

    with pytest.raises(ValueError, match="must be given together"):
        train_supcon(
            model,
            _make_projection_tail(),
            train_db,
            val_db,
            config,
            train_family_ids=family_ids[train_idx],
        )


def test_train_supcon_rejects_invalid_distance():
    train_db, val_db = _make_split_db(n=40)
    model = SupConEncoder(input_dim=5, encoder_hidden_dim=[8], latent_dim=3, seed=0)
    config = TrainConfig(epochs=1, batch_size=8, seed=0, device="cpu", distance="bogus")
    family_ids = np.zeros(40, dtype=np.int32)

    with pytest.raises(ValueError, match="distance must be one of"):
        train_supcon(
            model,
            _make_projection_tail(),
            train_db,
            val_db,
            config,
            train_family_ids=family_ids[:30],
            val_family_ids=family_ids[30:],
        )


def test_train_supcon_cosine_distance_returns_history():
    n = 40
    train_db, val_db = _make_split_db(n=n)
    train_idx, val_idx = _split_indices_like_train_val_split(n, 0.25, 0)
    rng = np.random.default_rng(0)
    family_ids = rng.integers(0, 3, size=n).astype(np.int32)

    model = SupConEncoder(input_dim=5, encoder_hidden_dim=[8], latent_dim=3, seed=0)
    config = TrainConfig(
        epochs=2, batch_size=8, seed=0, device="cpu", distance="cosine"
    )

    history = train_supcon(
        model,
        _make_projection_tail(),
        train_db,
        val_db,
        config,
        train_family_ids=family_ids[train_idx],
        val_family_ids=family_ids[val_idx],
    )

    assert len(history["train_loss"]) == 2
    assert all(math.isfinite(v) for v in history["train_loss"])
    assert all(math.isfinite(v) for v in history["val_loss"])


def test_train_supcon_invalid_batching_strategy():
    train_db, val_db = _make_split_db(n=40)
    model = SupConEncoder(input_dim=5, encoder_hidden_dim=[8], latent_dim=3, seed=0)
    config = TrainConfig(epochs=1, batch_size=8, seed=0, device="cpu")
    family_ids = np.zeros(40, dtype=np.int32)

    with pytest.raises(ValueError, match="batching_strategy must be one of"):
        train_supcon(
            model,
            _make_projection_tail(),
            train_db,
            val_db,
            config,
            train_family_ids=family_ids[:30],
            val_family_ids=family_ids[30:],
            batching_strategy="bogus",
        )


def test_train_supcon_balanced_batching_requires_family_ids():
    train_db, val_db = _make_split_db(n=40)
    model = SupConEncoder(input_dim=5, encoder_hidden_dim=[8], latent_dim=3, seed=0)
    config = TrainConfig(epochs=1, batch_size=8, seed=0, device="cpu")
    family_ids = np.zeros(40, dtype=np.int32)

    with pytest.raises(ValueError, match="requires batching_family_ids"):
        train_supcon(
            model,
            _make_projection_tail(),
            train_db,
            val_db,
            config,
            train_family_ids=family_ids[:30],
            val_family_ids=family_ids[30:],
            batching_strategy="balanced",
            batching_K=4,
        )


def test_train_supcon_balanced_batching_requires_positive_K():
    n = 40
    train_db, val_db = _make_split_db(n=n)
    train_idx, val_idx = _split_indices_like_train_val_split(n, 0.25, 0)
    model = SupConEncoder(input_dim=5, encoder_hidden_dim=[8], latent_dim=3, seed=0)
    config = TrainConfig(epochs=1, batch_size=8, seed=0, device="cpu")
    rng = np.random.default_rng(1)
    family_ids = rng.integers(0, 3, size=n).astype(np.int32)

    with pytest.raises(ValueError, match="batching_K must be a positive integer"):
        train_supcon(
            model,
            _make_projection_tail(),
            train_db,
            val_db,
            config,
            train_family_ids=family_ids[train_idx],
            val_family_ids=family_ids[val_idx],
            batching_strategy="balanced",
            batching_family_ids=family_ids[train_idx],
        )


def test_train_supcon_balanced_batching_requires_spacegroup_ids():
    n = 40
    train_db, val_db = _make_split_db(n=n)
    train_idx, val_idx = _split_indices_like_train_val_split(n, 0.25, 0)
    model = SupConEncoder(input_dim=5, encoder_hidden_dim=[8], latent_dim=3, seed=0)
    config = TrainConfig(epochs=1, batch_size=8, seed=0, device="cpu")
    rng = np.random.default_rng(1)
    family_ids = rng.integers(0, 3, size=n).astype(np.int32)

    # Spacegroup stratification is always applied for balanced batching now
    # (S=None means "use every spacegroup present", not "skip") -- so
    # batching_spacegroup_ids is required regardless of whether S is given.
    with pytest.raises(ValueError, match="requires batching_spacegroup_ids"):
        train_supcon(
            model,
            _make_projection_tail(),
            train_db,
            val_db,
            config,
            train_family_ids=family_ids[train_idx],
            val_family_ids=family_ids[val_idx],
            batching_strategy="balanced",
            batching_family_ids=family_ids[train_idx],
            batching_K=4,
        )


def test_train_supcon_balanced_batching_returns_history():
    n = 40
    train_db, val_db = _make_split_db(n=n)
    train_idx, val_idx = _split_indices_like_train_val_split(n, 0.25, 0)
    rng = np.random.default_rng(1)
    family_ids = rng.integers(0, 4, size=n).astype(np.int32)
    spacegroup_ids = rng.integers(0, 8, size=n).astype(np.int32)

    model = SupConEncoder(input_dim=5, encoder_hidden_dim=[8], latent_dim=3, seed=0)
    config = TrainConfig(epochs=2, batch_size=8, tau=0.1, seed=0, device="cpu")
    history = train_supcon(
        model,
        _make_projection_tail(),
        train_db,
        val_db,
        config,
        train_family_ids=family_ids[train_idx],
        val_family_ids=family_ids[val_idx],
        train_spacegroup_ids=spacegroup_ids[train_idx],
        val_spacegroup_ids=spacegroup_ids[val_idx],
        lambda_family=1.0,
        lambda_spacegroup=0.5,
        batching_strategy="balanced",
        batching_family_ids=family_ids[train_idx],
        batching_spacegroup_ids=spacegroup_ids[train_idx],
        batching_P=None,
        batching_K=4,
        batching_S=2,
    )

    for key in (
        "train_loss",
        "val_loss",
        "train_family_supcon",
        "val_family_supcon",
        "train_spacegroup_supcon",
        "val_spacegroup_supcon",
    ):
        assert key in history
        assert len(history[key]) == 2
        assert all(math.isfinite(v) for v in history[key])
        assert all(v >= 0.0 for v in history[key])


def test_train_supcon_balanced_batching_logs_warning_about_ignored_batch_size(caplog):
    n = 40
    train_db, val_db = _make_split_db(n=n)
    train_idx, val_idx = _split_indices_like_train_val_split(n, 0.25, 0)
    rng = np.random.default_rng(1)
    family_ids = rng.integers(0, 4, size=n).astype(np.int32)
    spacegroup_ids = rng.integers(0, 8, size=n).astype(np.int32)

    model = SupConEncoder(input_dim=5, encoder_hidden_dim=[8], latent_dim=3, seed=0)
    config = TrainConfig(epochs=1, batch_size=8, tau=0.1, seed=0, device="cpu")

    with caplog.at_level(logging.WARNING, logger="dim_red.pipeline"):
        train_supcon(
            model,
            _make_projection_tail(),
            train_db,
            val_db,
            config,
            train_family_ids=family_ids[train_idx],
            val_family_ids=family_ids[val_idx],
            batching_strategy="balanced",
            batching_family_ids=family_ids[train_idx],
            batching_spacegroup_ids=spacegroup_ids[train_idx],
            batching_K=5,
        )

    assert any("train.batch_size (8) is ignored" in r.message for r in caplog.records)


def test_train_supcon_balanced_batching_decoupled_from_loss_family_flag():
    """batching_family_ids can group batches by family even when the family
    SupCon loss term itself is inactive (mode == "spacegroup_only") --
    history must not contain a family term, only a spacegroup one.
    """
    n = 40
    train_db, val_db = _make_split_db(n=n)
    train_idx, val_idx = _split_indices_like_train_val_split(n, 0.25, 0)
    rng = np.random.default_rng(1)
    family_ids = rng.integers(0, 4, size=n).astype(np.int32)
    spacegroup_ids = rng.integers(0, 8, size=n).astype(np.int32)

    model = SupConEncoder(input_dim=5, encoder_hidden_dim=[8], latent_dim=3, seed=0)
    config = TrainConfig(epochs=1, batch_size=8, tau=0.1, seed=0, device="cpu")
    history = train_supcon(
        model,
        _make_projection_tail(),
        train_db,
        val_db,
        config,
        train_spacegroup_ids=spacegroup_ids[train_idx],
        val_spacegroup_ids=spacegroup_ids[val_idx],
        lambda_spacegroup=1.0,
        batching_strategy="balanced",
        batching_family_ids=family_ids[train_idx],
        batching_spacegroup_ids=spacegroup_ids[train_idx],
        batching_K=5,
    )

    assert "train_spacegroup_supcon" in history
    assert "train_family_supcon" not in history


def _family_ids_for(n, n_family=3, seed=1):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_family, size=n).astype(np.int32)


def test_train_supcon_early_stopping_disabled_runs_full_epochs():
    n = 40
    train_db, val_db = _make_split_db(n=n)
    train_idx, val_idx = _split_indices_like_train_val_split(n, 0.25, 0)
    family_ids = _family_ids_for(n)

    model = SupConEncoder(input_dim=5, encoder_hidden_dim=[8], latent_dim=3, seed=0)
    config = TrainConfig(epochs=4, batch_size=8, tau=0.1, seed=0, device="cpu")
    history = train_supcon(
        model,
        _make_projection_tail(),
        train_db,
        val_db,
        config,
        train_family_ids=family_ids[train_idx],
        val_family_ids=family_ids[val_idx],
    )
    assert len(history["train_loss"]) == 4


def test_train_supcon_early_stopping_stops_before_configured_epochs():
    n = 40
    train_db, val_db = _make_split_db(n=n)
    train_idx, val_idx = _split_indices_like_train_val_split(n, 0.25, 0)
    family_ids = _family_ids_for(n)

    model = SupConEncoder(input_dim=5, encoder_hidden_dim=[8], latent_dim=3, seed=0)
    config = TrainConfig(
        epochs=30,
        batch_size=8,
        tau=0.1,
        seed=0,
        device="cpu",
        early_stopping=True,
        early_stopping_patience=2,
    )
    history = train_supcon(
        model,
        _make_projection_tail(),
        train_db,
        val_db,
        config,
        train_family_ids=family_ids[train_idx],
        val_family_ids=family_ids[val_idx],
    )
    n_epochs_run = len(history["train_loss"])
    assert n_epochs_run < config.epochs
    best_epoch_idx = int(np.argmin(history["val_loss"]))
    assert best_epoch_idx <= n_epochs_run - 1 - config.early_stopping_patience


def test_train_supcon_early_stopping_restore_best_changes_final_params():
    n = 40
    train_db, val_db = _make_split_db(n=n)
    train_idx, val_idx = _split_indices_like_train_val_split(n, 0.25, 0)
    family_ids = _family_ids_for(n)

    model_restore = SupConEncoder(
        input_dim=5, encoder_hidden_dim=[8], latent_dim=3, seed=0
    )
    config_restore = TrainConfig(
        epochs=30,
        batch_size=8,
        tau=0.1,
        seed=0,
        device="cpu",
        early_stopping=True,
        early_stopping_patience=2,
        early_stopping_restore_best=True,
    )
    history_restore = train_supcon(
        model_restore,
        _make_projection_tail(),
        train_db,
        val_db,
        config_restore,
        train_family_ids=family_ids[train_idx],
        val_family_ids=family_ids[val_idx],
    )

    model_no_restore = SupConEncoder(
        input_dim=5, encoder_hidden_dim=[8], latent_dim=3, seed=0
    )
    config_no_restore = TrainConfig(
        epochs=30,
        batch_size=8,
        tau=0.1,
        seed=0,
        device="cpu",
        early_stopping=True,
        early_stopping_patience=2,
        early_stopping_restore_best=False,
    )
    history_no_restore = train_supcon(
        model_no_restore,
        _make_projection_tail(),
        train_db,
        val_db,
        config_no_restore,
        train_family_ids=family_ids[train_idx],
        val_family_ids=family_ids[val_idx],
    )

    assert history_restore["val_loss"] == history_no_restore["val_loss"]
    n_epochs_run = len(history_restore["val_loss"])
    assert n_epochs_run < config_restore.epochs

    best_epoch_idx = int(np.argmin(history_restore["val_loss"]))
    assert best_epoch_idx != n_epochs_run - 1

    leaves_restore = jax.tree_util.tree_leaves(model_restore.params)
    leaves_no_restore = jax.tree_util.tree_leaves(model_no_restore.params)
    assert any(
        not np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(leaves_restore, leaves_no_restore)
    )


def test_train_supcon_early_stopping_rejects_invalid_patience():
    n = 40
    train_db, val_db = _make_split_db(n=n)
    train_idx, val_idx = _split_indices_like_train_val_split(n, 0.25, 0)
    family_ids = _family_ids_for(n)

    with pytest.raises(ValueError, match="early_stopping_patience"):
        train_supcon(
            SupConEncoder(input_dim=5, encoder_hidden_dim=[8], latent_dim=3, seed=0),
            _make_projection_tail(),
            train_db,
            val_db,
            TrainConfig(
                epochs=1,
                batch_size=8,
                seed=0,
                device="cpu",
                early_stopping=True,
                early_stopping_patience=0,
            ),
            train_family_ids=family_ids[train_idx],
            val_family_ids=family_ids[val_idx],
        )


def test_train_supcon_early_stopping_rejects_negative_min_delta():
    n = 40
    train_db, val_db = _make_split_db(n=n)
    train_idx, val_idx = _split_indices_like_train_val_split(n, 0.25, 0)
    family_ids = _family_ids_for(n)

    with pytest.raises(ValueError, match="early_stopping_min_delta"):
        train_supcon(
            SupConEncoder(input_dim=5, encoder_hidden_dim=[8], latent_dim=3, seed=0),
            _make_projection_tail(),
            train_db,
            val_db,
            TrainConfig(
                epochs=1,
                batch_size=8,
                seed=0,
                device="cpu",
                early_stopping=True,
                early_stopping_min_delta=-1.0,
            ),
            train_family_ids=family_ids[train_idx],
            val_family_ids=family_ids[val_idx],
        )


def test_norm_penalty_matches_manual_computation():
    z = jnp.array([[3.0, 4.0], [0.0, 0.0], [1.0, 1.0]])
    # ||z_0||^2=25, ||z_1||^2=0, ||z_2||^2=2 -> mean = 27/3 = 9.
    assert float(norm_penalty(z)) == pytest.approx(9.0, abs=1e-6)


def test_norm_penalty_grows_with_embedding_scale():
    key = jax.random.PRNGKey(0)
    z = jax.random.normal(key, (10, 4))
    small = norm_penalty(z)
    large = norm_penalty(z * 10.0)
    assert float(large) == pytest.approx(float(small) * 100.0, rel=1e-4)


def test_train_supcon_lambda_norm_defaults_to_zero_and_is_backward_compatible():
    """lambda_norm=0.0 (the default) must leave train_loss/val_loss exactly
    equal to the family/spacegroup terms, same as before this feature
    existed -- norm_penalty is tracked but contributes nothing to the total.
    """
    n = 40
    train_db, val_db = _make_split_db(n=n)
    train_idx, val_idx = _split_indices_like_train_val_split(n, 0.25, 0)
    family_ids = _family_ids_for(n)

    model = SupConEncoder(input_dim=5, encoder_hidden_dim=[8], latent_dim=3, seed=0)
    config = TrainConfig(epochs=2, batch_size=8, tau=0.1, seed=0, device="cpu")
    history = train_supcon(
        model,
        _make_projection_tail(),
        train_db,
        val_db,
        config,
        train_family_ids=family_ids[train_idx],
        val_family_ids=family_ids[val_idx],
        lambda_family=1.0,
    )

    assert "train_norm_penalty" in history and "val_norm_penalty" in history
    assert all(v >= 0.0 for v in history["train_norm_penalty"])
    for total, family in zip(history["train_loss"], history["train_family_supcon"]):
        assert total == pytest.approx(family, abs=1e-6)


def test_train_supcon_lambda_norm_weights_total_correctly():
    n = 40
    train_db, val_db = _make_split_db(n=n)
    train_idx, val_idx = _split_indices_like_train_val_split(n, 0.25, 0)
    family_ids = _family_ids_for(n)

    model = SupConEncoder(input_dim=5, encoder_hidden_dim=[8], latent_dim=3, seed=0)
    config = TrainConfig(epochs=2, batch_size=8, tau=0.1, seed=0, device="cpu")
    history = train_supcon(
        model,
        _make_projection_tail(),
        train_db,
        val_db,
        config,
        train_family_ids=family_ids[train_idx],
        val_family_ids=family_ids[val_idx],
        lambda_family=1.0,
        lambda_norm=0.5,
    )

    for total, family, penalty in zip(
        history["train_loss"],
        history["train_family_supcon"],
        history["train_norm_penalty"],
    ):
        assert total == pytest.approx(family + 0.5 * penalty, abs=1e-4)
    for total, family, penalty in zip(
        history["val_loss"], history["val_family_supcon"], history["val_norm_penalty"]
    ):
        assert total == pytest.approx(family + 0.5 * penalty, abs=1e-4)


def test_train_supcon_norm_penalty_present_even_in_spacegroup_only_mode():
    """norm_penalty doesn't depend on labels at all, so it's tracked
    regardless of which SupCon terms (family/spacegroup) are active.
    """
    n = 40
    train_db, val_db = _make_split_db(n=n)
    train_idx, val_idx = _split_indices_like_train_val_split(n, 0.25, 0)
    rng = np.random.default_rng(1)
    spacegroup_ids = rng.integers(0, 6, size=n).astype(np.int32)

    model = SupConEncoder(input_dim=5, encoder_hidden_dim=[8], latent_dim=3, seed=0)
    config = TrainConfig(epochs=1, batch_size=8, tau=0.1, seed=0, device="cpu")
    history = train_supcon(
        model,
        _make_projection_tail(),
        train_db,
        val_db,
        config,
        train_spacegroup_ids=spacegroup_ids[train_idx],
        val_spacegroup_ids=spacegroup_ids[val_idx],
        lambda_spacegroup=1.0,
        lambda_norm=0.2,
    )

    assert "train_norm_penalty" in history and "val_norm_penalty" in history
    for total, spacegroup, penalty in zip(
        history["train_loss"],
        history["train_spacegroup_supcon"],
        history["train_norm_penalty"],
    ):
        assert total == pytest.approx(spacegroup + 0.2 * penalty, abs=1e-4)


def test_train_supcon_rejects_negative_lambda_norm():
    n = 40
    train_db, val_db = _make_split_db(n=n)
    train_idx, val_idx = _split_indices_like_train_val_split(n, 0.25, 0)
    family_ids = _family_ids_for(n)

    model = SupConEncoder(input_dim=5, encoder_hidden_dim=[8], latent_dim=3, seed=0)
    config = TrainConfig(epochs=1, batch_size=8, seed=0, device="cpu")

    with pytest.raises(ValueError, match="lambda_norm must be >= 0"):
        train_supcon(
            model,
            _make_projection_tail(),
            train_db,
            val_db,
            config,
            train_family_ids=family_ids[train_idx],
            val_family_ids=family_ids[val_idx],
            lambda_norm=-0.1,
        )


def test_train_supcon_computes_loss_on_projection_output_not_body_output():
    """projection_dim != latent_dim -- supcon_loss can only run on the
    projection tail's output (shape (batch, projection_dim)); if a
    regression made it run on the body's raw output (shape (batch,
    latent_dim)) instead, every anchor pair's positive/negative geometry
    would be computed in the wrong space and, since supcon_loss doesn't
    itself validate a specific width, the mismatch could only be caught by
    exercising an actual training step against two different widths -- this
    test's projection_dim (7) deliberately differs from latent_dim (3).
    """
    n = 40
    train_db, val_db = _make_split_db(n=n)
    train_idx, val_idx = _split_indices_like_train_val_split(n, 0.25, 0)
    family_ids = _family_ids_for(n)

    model = SupConEncoder(input_dim=5, encoder_hidden_dim=[8], latent_dim=3, seed=0)
    projection_tail = _make_projection_tail(latent_dim=3, projection_dim=7, seed=0)
    config = TrainConfig(epochs=1, batch_size=8, tau=0.1, seed=0, device="cpu")

    history = train_supcon(
        model,
        projection_tail,
        train_db,
        val_db,
        config,
        train_family_ids=family_ids[train_idx],
        val_family_ids=family_ids[val_idx],
    )

    assert len(history["train_loss"]) == 1
    assert projection_tail.project(np.zeros((1, 3), dtype=np.float32)).shape == (1, 7)


def test_train_supcon_mutates_both_body_and_projection_tail_params():
    n = 40
    train_db, val_db = _make_split_db(n=n)
    train_idx, val_idx = _split_indices_like_train_val_split(n, 0.25, 0)
    family_ids = _family_ids_for(n)

    model = SupConEncoder(input_dim=5, encoder_hidden_dim=[8], latent_dim=3, seed=0)
    projection_tail = _make_projection_tail(latent_dim=3, projection_dim=4, seed=0)
    body_params_before = jax.tree_util.tree_map(lambda a: np.array(a), model.params)
    tail_params_before = jax.tree_util.tree_map(
        lambda a: np.array(a), projection_tail.params
    )

    config = TrainConfig(epochs=3, batch_size=8, tau=0.1, seed=0, device="cpu")
    train_supcon(
        model,
        projection_tail,
        train_db,
        val_db,
        config,
        train_family_ids=family_ids[train_idx],
        val_family_ids=family_ids[val_idx],
    )

    body_changed = any(
        not np.array_equal(a, np.asarray(b))
        for a, b in zip(
            jax.tree_util.tree_leaves(body_params_before),
            jax.tree_util.tree_leaves(model.params),
        )
    )
    tail_changed = any(
        not np.array_equal(a, np.asarray(b))
        for a, b in zip(
            jax.tree_util.tree_leaves(tail_params_before),
            jax.tree_util.tree_leaves(projection_tail.params),
        )
    )
    assert body_changed
    assert tail_changed


def test_train_supcon_optimizer_velo_returns_history():
    """ "velo" still works now that it's no longer the default -- exercises
    the VeLO path (faked fast by tests/conftest.py's autouse fixture).
    """
    n = 40
    train_db, val_db = _make_split_db(n=n)
    train_idx, val_idx = _split_indices_like_train_val_split(n, 0.25, 0)
    family_ids = _family_ids_for(n)

    model = SupConEncoder(input_dim=5, encoder_hidden_dim=[8], latent_dim=3, seed=0)
    projection_tail = _make_projection_tail()
    config = TrainConfig(
        epochs=2, batch_size=8, tau=0.1, seed=0, device="cpu", optimizer="velo"
    )
    history = train_supcon(
        model,
        projection_tail,
        train_db,
        val_db,
        config,
        train_family_ids=family_ids[train_idx],
        val_family_ids=family_ids[val_idx],
    )

    assert len(history["train_loss"]) == 2
    assert all(math.isfinite(v) for v in history["train_loss"])


def test_train_supcon_rejects_invalid_optimizer():
    n = 40
    train_db, val_db = _make_split_db(n=n)
    train_idx, val_idx = _split_indices_like_train_val_split(n, 0.25, 0)
    family_ids = _family_ids_for(n)

    model = SupConEncoder(input_dim=5, encoder_hidden_dim=[8], latent_dim=3, seed=0)
    projection_tail = _make_projection_tail()
    config = TrainConfig(
        epochs=1, batch_size=8, seed=0, device="cpu", optimizer="bogus"
    )

    with pytest.raises(ValueError, match="optimizer must be one of"):
        train_supcon(
            model,
            projection_tail,
            train_db,
            val_db,
            config,
            train_family_ids=family_ids[train_idx],
            val_family_ids=family_ids[val_idx],
        )
