"""
Unit tests for phase-2 tail training (``dim_red.supcon.tail_training``):
training a classification or visualization tail directly on precomputed
representations, with no body forward pass at all.
"""

import logging
import math

import pytest

pytest.importorskip("jax")

import jax
import numpy as np

from dim_red.supcon.tail_training import (
    TailTrainConfig,
    train_classification_tail,
    train_visualization_tail,
)
from dim_red.supcon.tails import ClassificationTail, VisualizationTail


def _split_r_and_ids(n=40, input_dim=4, val_ratio=0.25, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.normal(size=(n, input_dim)).astype(np.float32)
    n_val = max(1, int(round(n * val_ratio)))
    indices = rng.permutation(n)
    train_idx, val_idx = indices[n_val:], indices[:n_val]
    return r, train_idx, val_idx


def _family_ids_for(n, n_family=3, seed=1):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_family, size=n).astype(np.int32)


def _spacegroup_ids_for(n, n_spacegroup=5, seed=2):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_spacegroup, size=n).astype(np.int32)


def _family_spacegroup_mask(family_ids, spacegroup_ids, n_family, n_spacegroup):
    mask = np.zeros((n_family, n_spacegroup), dtype=np.float32)
    mask[family_ids, spacegroup_ids] = 1.0
    return mask


# --- train_classification_tail ----------------------------------------------


def test_train_classification_tail_family_only_returns_history():
    n = 40
    r, train_idx, val_idx = _split_r_and_ids(n=n)
    family_ids = _family_ids_for(n)

    tail = ClassificationTail(input_dim=4, hidden_dim=8, n_family_classes=3, seed=0)
    config = TailTrainConfig(epochs=2, batch_size=8, seed=0, device="cpu")
    history = train_classification_tail(
        tail,
        r[train_idx],
        r[val_idx],
        config,
        train_family_ids=family_ids[train_idx],
        val_family_ids=family_ids[val_idx],
    )

    assert "train_family_ce" in history
    assert "train_spacegroup_ce" not in history
    for key in ("train_loss", "val_loss", "train_family_ce", "val_family_ce"):
        assert len(history[key]) == 2
        assert all(math.isfinite(v) for v in history[key])
    for total, family in zip(history["train_loss"], history["train_family_ce"]):
        assert total == pytest.approx(family, abs=1e-6)


def test_train_classification_tail_family_and_spacegroup_returns_history():
    n = 40
    r, train_idx, val_idx = _split_r_and_ids(n=n)
    family_ids = _family_ids_for(n, n_family=3)
    spacegroup_ids = _spacegroup_ids_for(n, n_spacegroup=5)
    mask = _family_spacegroup_mask(family_ids, spacegroup_ids, 3, 5)

    tail = ClassificationTail(
        input_dim=4, hidden_dim=8, n_family_classes=3, n_spacegroup_classes=5, seed=0
    )
    config = TailTrainConfig(epochs=2, batch_size=8, seed=0, device="cpu")
    history = train_classification_tail(
        tail,
        r[train_idx],
        r[val_idx],
        config,
        train_family_ids=family_ids[train_idx],
        val_family_ids=family_ids[val_idx],
        train_spacegroup_ids=spacegroup_ids[train_idx],
        val_spacegroup_ids=spacegroup_ids[val_idx],
        lambda_family=1.0,
        lambda_spacegroup=0.5,
        family_spacegroup_mask=mask,
    )

    for key in (
        "train_loss",
        "val_loss",
        "train_family_ce",
        "val_family_ce",
        "train_spacegroup_ce",
        "val_spacegroup_ce",
    ):
        assert key in history
        assert len(history[key]) == 2
        assert all(math.isfinite(v) for v in history[key])


def test_train_classification_tail_requires_at_least_one_label_type():
    r, train_idx, val_idx = _split_r_and_ids(n=40)
    tail = ClassificationTail(input_dim=4, hidden_dim=8, n_family_classes=3, seed=0)
    config = TailTrainConfig(epochs=1, batch_size=8, seed=0, device="cpu")

    with pytest.raises(ValueError, match="nothing to classify"):
        train_classification_tail(tail, r[train_idx], r[val_idx], config)


def test_train_classification_tail_spacegroup_requires_family_and_mask():
    n = 40
    r, train_idx, val_idx = _split_r_and_ids(n=n)
    spacegroup_ids = _spacegroup_ids_for(n)

    tail = ClassificationTail(
        input_dim=4, hidden_dim=8, n_family_classes=3, n_spacegroup_classes=5, seed=0
    )
    config = TailTrainConfig(epochs=1, batch_size=8, seed=0, device="cpu")

    with pytest.raises(ValueError, match="require train_family_ids"):
        train_classification_tail(
            tail,
            r[train_idx],
            r[val_idx],
            config,
            train_spacegroup_ids=spacegroup_ids[train_idx],
            val_spacegroup_ids=spacegroup_ids[val_idx],
        )


def test_train_classification_tail_mutates_tail_params():
    n = 40
    r, train_idx, val_idx = _split_r_and_ids(n=n)
    family_ids = _family_ids_for(n)

    tail = ClassificationTail(input_dim=4, hidden_dim=8, n_family_classes=3, seed=0)
    params_before = jax.tree_util.tree_map(lambda a: np.array(a), tail.params)
    config = TailTrainConfig(epochs=3, batch_size=8, seed=0, device="cpu")
    train_classification_tail(
        tail,
        r[train_idx],
        r[val_idx],
        config,
        train_family_ids=family_ids[train_idx],
        val_family_ids=family_ids[val_idx],
    )
    changed = any(
        not np.array_equal(a, np.asarray(b))
        for a, b in zip(
            jax.tree_util.tree_leaves(params_before),
            jax.tree_util.tree_leaves(tail.params),
        )
    )
    assert changed


def test_train_classification_tail_early_stopping_stops_before_configured_epochs():
    n = 40
    r, train_idx, val_idx = _split_r_and_ids(n=n)
    family_ids = _family_ids_for(n)

    tail = ClassificationTail(input_dim=4, hidden_dim=8, n_family_classes=3, seed=0)
    config = TailTrainConfig(
        epochs=30,
        batch_size=8,
        seed=0,
        device="cpu",
        # Tuned against VeLO's convergence timing (faked fast by
        # tests/conftest.py's fixture) -- this test is about early
        # stopping's own bookkeeping, not optimizer choice.
        optimizer="velo",
        early_stopping=True,
        early_stopping_patience=2,
    )
    history = train_classification_tail(
        tail,
        r[train_idx],
        r[val_idx],
        config,
        train_family_ids=family_ids[train_idx],
        val_family_ids=family_ids[val_idx],
    )
    assert len(history["train_loss"]) < config.epochs


# --- train_visualization_tail ------------------------------------------------


@pytest.mark.parametrize("output_dim", [2, 3])
def test_train_visualization_tail_random_batching_returns_history(output_dim):
    n = 40
    r, train_idx, val_idx = _split_r_and_ids(n=n)
    family_ids = _family_ids_for(n)

    tail = VisualizationTail(input_dim=4, hidden_dim=[8], output_dim=output_dim, seed=0)
    config = TailTrainConfig(epochs=2, batch_size=8, tau=0.1, seed=0, device="cpu")
    history = train_visualization_tail(
        tail,
        r[train_idx],
        r[val_idx],
        config,
        train_family_ids=family_ids[train_idx],
        val_family_ids=family_ids[val_idx],
    )

    for key in (
        "train_loss",
        "val_loss",
        "train_norm_penalty",
        "val_norm_penalty",
        "train_family_supcon",
        "val_family_supcon",
    ):
        assert key in history
        assert len(history[key]) == 2
        assert all(math.isfinite(v) for v in history[key])
    assert tail.project(r[:1]).shape == (1, output_dim)


def test_train_visualization_tail_balanced_batching_returns_history():
    n = 40
    r, train_idx, val_idx = _split_r_and_ids(n=n)
    family_ids = _family_ids_for(n, n_family=4)
    spacegroup_ids = _spacegroup_ids_for(n, n_spacegroup=8)

    tail = VisualizationTail(input_dim=4, hidden_dim=[8], output_dim=2, seed=0)
    config = TailTrainConfig(epochs=2, batch_size=8, tau=0.1, seed=0, device="cpu")
    history = train_visualization_tail(
        tail,
        r[train_idx],
        r[val_idx],
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
        batching_K=4,
        batching_S=2,
    )

    for key in (
        "train_family_supcon",
        "val_family_supcon",
        "train_spacegroup_supcon",
        "val_spacegroup_supcon",
    ):
        assert key in history
        assert len(history[key]) == 2


def test_train_visualization_tail_balanced_batching_logs_warning(caplog):
    n = 40
    r, train_idx, val_idx = _split_r_and_ids(n=n)
    family_ids = _family_ids_for(n, n_family=4)
    spacegroup_ids = _spacegroup_ids_for(n, n_spacegroup=8)

    tail = VisualizationTail(input_dim=4, hidden_dim=[8], output_dim=2, seed=0)
    config = TailTrainConfig(epochs=1, batch_size=8, tau=0.1, seed=0, device="cpu")

    with caplog.at_level(logging.WARNING, logger="dim_red.pipeline"):
        train_visualization_tail(
            tail,
            r[train_idx],
            r[val_idx],
            config,
            train_family_ids=family_ids[train_idx],
            val_family_ids=family_ids[val_idx],
            batching_strategy="balanced",
            batching_family_ids=family_ids[train_idx],
            batching_spacegroup_ids=spacegroup_ids[train_idx],
            batching_K=5,
        )

    assert any("batch_size (8) is ignored" in rec.message for rec in caplog.records)


def test_train_visualization_tail_requires_at_least_one_label_type():
    r, train_idx, val_idx = _split_r_and_ids(n=40)
    tail = VisualizationTail(input_dim=4, hidden_dim=[8], output_dim=2, seed=0)
    config = TailTrainConfig(epochs=1, batch_size=8, seed=0, device="cpu")

    with pytest.raises(ValueError, match="nothing to contrast on"):
        train_visualization_tail(tail, r[train_idx], r[val_idx], config)


def test_train_visualization_tail_lambda_norm_weights_total_correctly():
    n = 40
    r, train_idx, val_idx = _split_r_and_ids(n=n)
    family_ids = _family_ids_for(n)

    tail = VisualizationTail(input_dim=4, hidden_dim=[8], output_dim=2, seed=0)
    config = TailTrainConfig(epochs=2, batch_size=8, tau=0.1, seed=0, device="cpu")
    history = train_visualization_tail(
        tail,
        r[train_idx],
        r[val_idx],
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


def test_train_visualization_tail_mutates_tail_params():
    n = 40
    r, train_idx, val_idx = _split_r_and_ids(n=n)
    family_ids = _family_ids_for(n)

    tail = VisualizationTail(input_dim=4, hidden_dim=[8], output_dim=2, seed=0)
    params_before = jax.tree_util.tree_map(lambda a: np.array(a), tail.params)
    config = TailTrainConfig(epochs=3, batch_size=8, tau=0.1, seed=0, device="cpu")
    train_visualization_tail(
        tail,
        r[train_idx],
        r[val_idx],
        config,
        train_family_ids=family_ids[train_idx],
        val_family_ids=family_ids[val_idx],
    )
    changed = any(
        not np.array_equal(a, np.asarray(b))
        for a, b in zip(
            jax.tree_util.tree_leaves(params_before),
            jax.tree_util.tree_leaves(tail.params),
        )
    )
    assert changed


def test_train_visualization_tail_early_stopping_stops_before_configured_epochs():
    n = 40
    r, train_idx, val_idx = _split_r_and_ids(n=n)
    family_ids = _family_ids_for(n)

    tail = VisualizationTail(input_dim=4, hidden_dim=[8], output_dim=2, seed=0)
    config = TailTrainConfig(
        epochs=30,
        batch_size=8,
        tau=0.1,
        seed=0,
        device="cpu",
        early_stopping=True,
        early_stopping_patience=2,
    )
    history = train_visualization_tail(
        tail,
        r[train_idx],
        r[val_idx],
        config,
        train_family_ids=family_ids[train_idx],
        val_family_ids=family_ids[val_idx],
    )
    assert len(history["train_loss"]) < config.epochs


def test_train_visualization_tail_rejects_invalid_distance():
    r, train_idx, val_idx = _split_r_and_ids(n=40)
    family_ids = _family_ids_for(40)
    tail = VisualizationTail(input_dim=4, hidden_dim=[8], output_dim=2, seed=0)
    config = TailTrainConfig(
        epochs=1, batch_size=8, seed=0, device="cpu", distance="bogus"
    )

    with pytest.raises(ValueError, match="distance must be one of"):
        train_visualization_tail(
            tail,
            r[train_idx],
            r[val_idx],
            config,
            train_family_ids=family_ids[train_idx],
            val_family_ids=family_ids[val_idx],
        )


def test_train_classification_tail_optimizer_velo_returns_history():
    """ "velo" still works now that it's no longer the default -- exercises
    the VeLO path (faked fast by tests/conftest.py's autouse fixture).
    """
    n = 40
    r, train_idx, val_idx = _split_r_and_ids(n=n)
    family_ids = _family_ids_for(n)

    tail = ClassificationTail(input_dim=4, hidden_dim=8, n_family_classes=3, seed=0)
    config = TailTrainConfig(
        epochs=2, batch_size=8, seed=0, device="cpu", optimizer="velo"
    )
    history = train_classification_tail(
        tail,
        r[train_idx],
        r[val_idx],
        config,
        train_family_ids=family_ids[train_idx],
        val_family_ids=family_ids[val_idx],
    )
    assert len(history["train_loss"]) == 2
    assert all(math.isfinite(v) for v in history["train_loss"])


def test_train_classification_tail_rejects_invalid_optimizer():
    n = 40
    r, train_idx, val_idx = _split_r_and_ids(n=n)
    family_ids = _family_ids_for(n)

    tail = ClassificationTail(input_dim=4, hidden_dim=8, n_family_classes=3, seed=0)
    config = TailTrainConfig(
        epochs=1, batch_size=8, seed=0, device="cpu", optimizer="bogus"
    )

    with pytest.raises(ValueError, match="optimizer must be one of"):
        train_classification_tail(
            tail,
            r[train_idx],
            r[val_idx],
            config,
            train_family_ids=family_ids[train_idx],
            val_family_ids=family_ids[val_idx],
        )


def test_train_visualization_tail_optimizer_velo_returns_history():
    """ "velo" still works now that it's no longer the default -- exercises
    the VeLO path (faked fast by tests/conftest.py's autouse fixture).
    """
    n = 40
    r, train_idx, val_idx = _split_r_and_ids(n=n)
    family_ids = _family_ids_for(n)

    tail = VisualizationTail(input_dim=4, hidden_dim=[8], output_dim=2, seed=0)
    config = TailTrainConfig(
        epochs=2, batch_size=8, tau=0.1, seed=0, device="cpu", optimizer="velo"
    )
    history = train_visualization_tail(
        tail,
        r[train_idx],
        r[val_idx],
        config,
        train_family_ids=family_ids[train_idx],
        val_family_ids=family_ids[val_idx],
    )
    assert len(history["train_loss"]) == 2
    assert all(math.isfinite(v) for v in history["train_loss"])


def test_train_visualization_tail_rejects_invalid_optimizer():
    r, train_idx, val_idx = _split_r_and_ids(n=40)
    family_ids = _family_ids_for(40)
    tail = VisualizationTail(input_dim=4, hidden_dim=[8], output_dim=2, seed=0)
    config = TailTrainConfig(
        epochs=1, batch_size=8, seed=0, device="cpu", optimizer="bogus"
    )

    with pytest.raises(ValueError, match="optimizer must be one of"):
        train_visualization_tail(
            tail,
            r[train_idx],
            r[val_idx],
            config,
            train_family_ids=family_ids[train_idx],
            val_family_ids=family_ids[val_idx],
        )
