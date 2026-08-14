"""
Unit tests for dim_red.cgcnn.training.train_cgcnn.
"""

import math

import pytest

pytest.importorskip("jax")

import numpy as np
from ase import Atoms

from dim_red.cgcnn.database import GraphDatabase
from dim_red.cgcnn.graph import atoms_list_to_graph_arrays
from dim_red.cgcnn.model import CGCNNEncoder
from dim_red.cgcnn.training import TrainConfig, train_cgcnn


def _triangle(jitter, seed):
    rng = np.random.default_rng(seed)
    pos = np.array([[0, 0, 0], [1.0, 0, 0], [0.5, 0.87, 0]]) + rng.normal(
        scale=jitter, size=(3, 3)
    )
    return Atoms("XXX", positions=pos, cell=[10, 10, 10], pbc=False)


def _square(jitter, seed):
    rng = np.random.default_rng(1000 + seed)
    pos = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]) + rng.normal(
        scale=jitter, size=(4, 3)
    )
    return Atoms("XXXX", positions=pos, cell=[10, 10, 10], pbc=False)


def _make_split_db(n_per_class=16, seed=0):
    atoms_list = [_triangle(0.05, i) for i in range(n_per_class)] + [
        _square(0.05, i) for i in range(n_per_class)
    ]
    family_ids = np.array([0] * n_per_class + [1] * n_per_class, dtype=np.int32)
    spacegroup_ids = np.array([0] * n_per_class + [1] * n_per_class, dtype=np.int32)
    arrays = atoms_list_to_graph_arrays(
        atoms_list, radius=3.0, max_num_nbr=4, max_species=1
    )
    db = GraphDatabase.from_arrays(*arrays)

    rng = np.random.default_rng(seed)
    n = db.n_samples
    idx = rng.permutation(n)
    n_val = max(1, n // 5)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    n_gaussian = arrays[2].shape[-1]
    return (
        db[train_idx],
        db[val_idx],
        family_ids[train_idx],
        family_ids[val_idx],
        spacegroup_ids[train_idx],
        spacegroup_ids[val_idx],
        n_gaussian,
    )


def _tiny_model(n_gaussian, **kwargs):
    defaults = dict(
        atom_fea_len=8,
        n_conv=2,
        h_fea_len=8,
        n_h=1,
        latent_dim=4,
        n_gaussian=n_gaussian,
        max_species=1,
        seed=0,
    )
    defaults.update(kwargs)
    return CGCNNEncoder(**defaults)


def test_train_cgcnn_returns_history_with_family_and_spacegroup():
    (
        train_db,
        val_db,
        train_family,
        val_family,
        train_sg,
        val_sg,
        n_gaussian,
    ) = _make_split_db()
    model = _tiny_model(n_gaussian, n_family_classes=2, n_spacegroup_classes=2)
    config = TrainConfig(epochs=3, batch_size=8, seed=0, device="cpu")
    mask = np.ones((2, 2), dtype=np.float32)

    history = train_cgcnn(
        model,
        train_db,
        val_db,
        config,
        train_family_ids=train_family,
        val_family_ids=val_family,
        train_spacegroup_ids=train_sg,
        val_spacegroup_ids=val_sg,
        family_spacegroup_mask=mask,
    )

    for key in (
        "train_loss",
        "train_family_ce",
        "val_loss",
        "val_family_ce",
        "train_spacegroup_ce",
        "val_spacegroup_ce",
    ):
        assert key in history
        assert len(history[key]) == 3
        assert all(math.isfinite(v) for v in history[key])
        assert all(v >= 0.0 for v in history[key])


def test_train_cgcnn_family_only_has_no_spacegroup_keys():
    (
        train_db,
        val_db,
        train_family,
        val_family,
        _train_sg,
        _val_sg,
        n_gaussian,
    ) = _make_split_db()
    model = _tiny_model(n_gaussian, n_family_classes=2)
    config = TrainConfig(epochs=2, batch_size=8, seed=0, device="cpu")

    history = train_cgcnn(
        model,
        train_db,
        val_db,
        config,
        train_family_ids=train_family,
        val_family_ids=val_family,
    )
    assert "train_spacegroup_ce" not in history
    assert "val_spacegroup_ce" not in history


def test_train_cgcnn_requires_family_ids():
    (train_db, val_db, *_rest, n_gaussian) = _make_split_db()
    model = _tiny_model(n_gaussian, n_family_classes=2)
    config = TrainConfig(epochs=2, batch_size=8, seed=0, device="cpu")
    with pytest.raises(
        ValueError, match="train_family_ids and val_family_ids are required"
    ):
        train_cgcnn(
            model, train_db, val_db, config, train_family_ids=None, val_family_ids=None
        )


def test_train_cgcnn_spacegroup_requires_mask():
    (
        train_db,
        val_db,
        train_family,
        val_family,
        train_sg,
        val_sg,
        n_gaussian,
    ) = _make_split_db()
    model = _tiny_model(n_gaussian, n_family_classes=2, n_spacegroup_classes=2)
    config = TrainConfig(epochs=2, batch_size=8, seed=0, device="cpu")
    with pytest.raises(ValueError, match="family_spacegroup_mask"):
        train_cgcnn(
            model,
            train_db,
            val_db,
            config,
            train_family_ids=train_family,
            val_family_ids=val_family,
            train_spacegroup_ids=train_sg,
            val_spacegroup_ids=val_sg,
        )


def test_train_cgcnn_early_stopping_shortens_history():
    (
        train_db,
        val_db,
        train_family,
        val_family,
        _train_sg,
        _val_sg,
        n_gaussian,
    ) = _make_split_db()
    model = _tiny_model(n_gaussian, n_family_classes=2)
    config = TrainConfig(
        epochs=30,
        batch_size=8,
        seed=0,
        device="cpu",
        early_stopping=True,
        early_stopping_patience=2,
    )
    history = train_cgcnn(
        model,
        train_db,
        val_db,
        config,
        train_family_ids=train_family,
        val_family_ids=val_family,
    )
    n_epochs_run = len(history["train_loss"])
    assert n_epochs_run <= config.epochs
    best_epoch_idx = int(np.argmin(history["val_loss"]))
    assert best_epoch_idx <= n_epochs_run - 1


def test_train_cgcnn_loss_decreases_on_separable_toy_problem():
    (
        train_db,
        val_db,
        train_family,
        val_family,
        _train_sg,
        _val_sg,
        n_gaussian,
    ) = _make_split_db(n_per_class=20)
    model = _tiny_model(
        n_gaussian, atom_fea_len=16, h_fea_len=16, latent_dim=8, n_family_classes=2
    )
    config = TrainConfig(
        epochs=25, batch_size=8, seed=0, device="cpu", learning_rate=5e-3
    )
    history = train_cgcnn(
        model,
        train_db,
        val_db,
        config,
        train_family_ids=train_family,
        val_family_ids=val_family,
    )
    assert history["train_family_ce"][-1] < history["train_family_ce"][0]

    z_val = np.asarray(model.encode(val_db.as_tuple()))
    preds = np.asarray(model.classify_family(z_val)).argmax(axis=1)
    acc = (preds == val_family).mean()
    assert acc > 0.7
