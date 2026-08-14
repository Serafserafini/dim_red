"""
Unit tests for dim_red.cgcnn.database.GraphDatabase.
"""

import numpy as np
import pytest

from dim_red.cgcnn.database import GraphDatabase


def _make_arrays(n=10, max_atoms=6, max_num_nbr=4, n_gaussian=5):
    rng = np.random.default_rng(0)
    local_species_idx = rng.integers(0, 3, size=(n, max_atoms)).astype(np.int32)
    nbr_idx = rng.integers(0, max_atoms, size=(n, max_atoms, max_num_nbr)).astype(
        np.int32
    )
    nbr_fea = rng.random((n, max_atoms, max_num_nbr, n_gaussian)).astype(np.float32)
    nbr_mask = rng.integers(0, 2, size=(n, max_atoms, max_num_nbr)).astype(np.float32)
    atom_mask = rng.integers(0, 2, size=(n, max_atoms)).astype(np.float32)
    return local_species_idx, nbr_idx, nbr_fea, nbr_mask, atom_mask


def test_graph_database_from_arrays_shapes():
    arrays = _make_arrays(n=10, max_atoms=6, max_num_nbr=4, n_gaussian=5)
    db = GraphDatabase.from_arrays(*arrays)
    assert db.n_samples == 10
    assert db.local_species_idx.shape == (10, 6)
    assert db.nbr_idx.shape == (10, 6, 4)
    assert db.nbr_fea.shape == (10, 6, 4, 5)
    assert db.nbr_mask.shape == (10, 6, 4)
    assert db.atom_mask.shape == (10, 6)
    assert db.local_species_idx.dtype == np.int32
    assert db.nbr_fea.dtype == np.float32


@pytest.mark.parametrize("ndim_error_field", ["local_species_idx", "atom_mask"])
def test_graph_database_from_arrays_validates_rank(ndim_error_field):
    arrays = list(_make_arrays())
    idx = {"local_species_idx": 0, "atom_mask": 4}[ndim_error_field]
    arrays[idx] = arrays[idx][0]  # drop the leading dim -> wrong rank
    with pytest.raises(ValueError):
        GraphDatabase.from_arrays(*arrays)


def test_graph_database_from_arrays_validates_shape_consistency():
    local_species_idx, nbr_idx, nbr_fea, nbr_mask, atom_mask = _make_arrays()
    bad_nbr_fea = nbr_fea[:, :, :2, :]  # inconsistent max_num_nbr
    with pytest.raises(ValueError, match="inconsistent with"):
        GraphDatabase.from_arrays(
            local_species_idx, nbr_idx, bad_nbr_fea, nbr_mask, atom_mask
        )


def test_graph_database_as_tuple_order():
    arrays = _make_arrays()
    db = GraphDatabase.from_arrays(*arrays)
    t = db.as_tuple()
    assert len(t) == 5
    assert t[0] is db.local_species_idx
    assert t[4] is db.atom_mask
    assert all(a.shape[0] == db.n_samples for a in t)


def test_graph_database_getitem_row_indexes_all_arrays():
    arrays = _make_arrays(n=10)
    db = GraphDatabase.from_arrays(*arrays)
    idx = np.array([0, 3, 7])
    sub = db[idx]
    assert sub.n_samples == 3
    np.testing.assert_array_equal(sub.local_species_idx, db.local_species_idx[idx])
    np.testing.assert_array_equal(sub.atom_mask, db.atom_mask[idx])


def test_graph_database_train_val_split_sizes():
    arrays = _make_arrays(n=20)
    db = GraphDatabase.from_arrays(*arrays)
    train_db, val_db = db.train_val_split(val_ratio=0.25, seed=0)
    assert train_db.n_samples + val_db.n_samples == 20
    assert val_db.n_samples == 5


def test_graph_database_train_val_split_at_least_one_per_side():
    arrays = _make_arrays(n=2)
    db = GraphDatabase.from_arrays(*arrays)
    train_db, val_db = db.train_val_split(val_ratio=0.01, seed=0)
    assert train_db.n_samples >= 1
    assert val_db.n_samples >= 1


def test_graph_database_train_val_split_invalid_ratio():
    arrays = _make_arrays()
    db = GraphDatabase.from_arrays(*arrays)
    with pytest.raises(ValueError, match="open interval"):
        db.train_val_split(val_ratio=1.5, seed=0)


def test_graph_database_train_val_split_reproducible():
    arrays = _make_arrays(n=20)
    db = GraphDatabase.from_arrays(*arrays)
    train_db1, val_db1 = db.train_val_split(val_ratio=0.2, seed=7)
    train_db2, val_db2 = db.train_val_split(val_ratio=0.2, seed=7)
    np.testing.assert_array_equal(
        train_db1.local_species_idx, train_db2.local_species_idx
    )
    np.testing.assert_array_equal(val_db1.local_species_idx, val_db2.local_species_idx)
