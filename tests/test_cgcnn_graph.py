"""
Unit tests for CGCNN graph construction (dim_red.cgcnn.graph) -- pure
NumPy/ASE, no jax dependency, so no pytest.importorskip("jax") needed here.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from dim_red.cgcnn.graph import (
    _local_species_indices,
    atoms_list_to_graph_arrays,
    atoms_to_graph,
)

# --- _local_species_indices (the species-agnostic remapping) ---------------


def test_local_species_indices_is_chemistry_agnostic():
    """Two structures with the same positional pattern of "lower atomic
    number here, higher atomic number there" must produce IDENTICAL local
    species indices regardless of which real elements are involved -- the
    core regression test for the model never learning real chemistry.
    """
    o_fe = np.array([8, 26, 8, 26])  # O(8) < Fe(26)
    na_cl = np.array([11, 17, 11, 17])  # Na(11) < Cl(17)
    idx_o_fe = _local_species_indices(o_fe, max_species=10)
    idx_na_cl = _local_species_indices(na_cl, max_species=10)
    np.testing.assert_array_equal(idx_o_fe, idx_na_cl)


def test_local_species_indices_single_species():
    atomic_numbers = np.array([29, 29, 29, 29])
    idx = _local_species_indices(atomic_numbers, max_species=10)
    np.testing.assert_array_equal(idx, np.ones(4, dtype=np.int32))


def test_local_species_indices_zero_reserved_for_padding():
    atomic_numbers = np.array([8, 26])
    idx = _local_species_indices(atomic_numbers, max_species=10)
    assert 0 not in idx


def test_local_species_indices_raises_when_more_species_than_max_species():
    atomic_numbers = np.arange(1, 12)  # 11 distinct species
    with pytest.raises(ValueError, match="exceeding max_species"):
        _local_species_indices(atomic_numbers, max_species=10)


# --- atoms_to_graph / neighbor search ---------------------------------------


def test_atoms_to_graph_periodic_beyond_naive_mic():
    """Cu FCC cubic cell (lattice constant ~3.6 A) with a cutoff radius
    (8.0, CGCNN's own default) far larger than half the cell's width --
    naive minimum-image-convention would miss most neighbors; the correct
    answer (ase.neighborlist, cell-shift aware) finds all 12 FCC nearest
    neighbors for every atom.
    """
    atoms = bulk("Cu", cubic=True)
    graph = atoms_to_graph(atoms, radius=8.0, max_num_nbr=12)
    assert graph["nbr_mask"].sum(axis=1).tolist() == [12.0] * len(atoms)


def test_atoms_to_graph_shapes():
    atoms = bulk("Cu", cubic=True)
    graph = atoms_to_graph(atoms, radius=5.0, max_num_nbr=8, step=0.2)
    n_atoms = len(atoms)
    n_gaussian = int(round(5.0 / 0.2)) + 1
    assert graph["local_species_idx"].shape == (n_atoms,)
    assert graph["nbr_idx"].shape == (n_atoms, 8)
    assert graph["nbr_fea"].shape == (n_atoms, 8, n_gaussian)
    assert graph["nbr_mask"].shape == (n_atoms, 8)


def test_gaussian_expansion_bounds():
    atoms = bulk("Cu", cubic=True)
    graph = atoms_to_graph(atoms, radius=5.0, max_num_nbr=8)
    assert graph["nbr_fea"].min() >= 0.0
    assert graph["nbr_fea"].max() <= 1.0 + 1e-6


def test_neighbor_list_truncates_to_max_num_nbr():
    atoms = bulk("Cu", cubic=True)
    graph = atoms_to_graph(atoms, radius=8.0, max_num_nbr=4)
    assert graph["nbr_mask"].sum(axis=1).tolist() == [4.0] * len(atoms)


def test_neighbor_list_pads_when_fewer_than_max_num_nbr():
    """Cu FCC's 12 nearest neighbors are all equidistant (~2.55 A); asking
    for more slots than that finds a nonzero count still short of
    max_num_nbr, so the remainder is masked out."""
    atoms = bulk("Cu", cubic=True)
    graph = atoms_to_graph(atoms, radius=2.6, max_num_nbr=20)
    counts = graph["nbr_mask"].sum(axis=1)
    assert (counts == 12).all()
    assert (counts < 20).all()


def test_atoms_to_graph_nonperiodic():
    mol = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 0.96], [0.93, 0, -0.24]], pbc=False)
    graph = atoms_to_graph(mol, radius=2.0, max_num_nbr=4)
    assert graph["nbr_mask"].sum() > 0


# --- atoms_list_to_graph_arrays (whole-dataset padding) ---------------------


def test_atoms_list_to_graph_arrays_pads_to_dataset_max_atoms():
    small = bulk("Cu", cubic=True)
    large = bulk("Cu", cubic=True) * (2, 1, 1)
    arrays = atoms_list_to_graph_arrays(
        [small, large], radius=5.0, max_num_nbr=8, max_species=2
    )
    local_species_idx, nbr_idx, nbr_fea, nbr_mask, atom_mask = arrays
    assert local_species_idx.shape[1] == len(large)
    assert atom_mask[0].sum() == len(small)
    assert atom_mask[1].sum() == len(large)
    # Padding rows are exactly zeroed out.
    assert (local_species_idx[0, len(small) :] == 0).all()


def test_atoms_list_to_graph_arrays_raises_if_explicit_max_atoms_too_small():
    small = bulk("Cu", cubic=True)
    large = bulk("Cu", cubic=True) * (2, 1, 1)
    with pytest.raises(ValueError, match="smaller than the largest structure"):
        atoms_list_to_graph_arrays(
            [small, large], radius=5.0, max_num_nbr=8, max_atoms=len(small)
        )


def test_atoms_list_to_graph_arrays_raises_on_empty_list():
    with pytest.raises(ValueError, match="must not be empty"):
        atoms_list_to_graph_arrays([])


def test_atoms_list_to_graph_arrays_larger_max_atoms_does_not_affect_real_rows():
    """max_atoms is purely an array-shape convenience -- extra padding rows
    must not perturb the real atoms' features (see the model-level
    invariance test in test_cgcnn_model.py for the trained-encoder version
    of this property)."""
    atoms = [bulk("Cu", cubic=True)]
    tight = atoms_list_to_graph_arrays(atoms, radius=5.0, max_num_nbr=8, max_species=2)
    padded = atoms_list_to_graph_arrays(
        atoms, radius=5.0, max_num_nbr=8, max_species=2, max_atoms=10
    )
    n_real = len(atoms[0])
    np.testing.assert_array_equal(
        tight[0][0], padded[0][0, :n_real]
    )  # local_species_idx
    np.testing.assert_array_equal(tight[2][0], padded[2][0, :n_real])  # nbr_fea
