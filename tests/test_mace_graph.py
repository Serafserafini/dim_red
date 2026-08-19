"""
Unit tests for dim_red.mace.graph -- pure NumPy/ASE, no mace_jax dependency
at all, so these run unconditionally (unlike dim_red.mace.model, which
requires mace_jax only once MaceEncoder is actually instantiated).
"""

import numpy as np
from ase import Atoms

from dim_red.mace.graph import atoms_list_to_mace_batch, atoms_to_mace_graph


def _fake_structure(symbol="Cu", n=2):
    return Atoms(
        symbol * n,
        positions=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]][:n],
        cell=[4.0, 4.0, 4.0],
        pbc=True,
    )


def test_atoms_to_mace_graph_shapes():
    atoms = _fake_structure("Cu", 2)
    graph = atoms_to_mace_graph(atoms, r_max=3.0)

    assert graph["positions"].shape == (2, 3)
    assert graph["atomic_numbers"].shape == (2,)
    assert graph["cell"].shape == (3, 3)
    assert graph["edge_index"].shape[0] == 2
    n_edges = graph["edge_index"].shape[1]
    assert graph["shifts"].shape == (n_edges, 3)


def test_atoms_to_mace_graph_uses_real_atomic_numbers():
    atoms = _fake_structure("Cu", 2)
    graph = atoms_to_mace_graph(atoms, r_max=3.0)
    np.testing.assert_array_equal(graph["atomic_numbers"], [29, 29])


def test_atoms_to_mace_graph_finds_periodic_neighbors():
    # Two Cu atoms 1.5 A apart in a 4 A cubic cell -- well within a 3.0 A
    # cutoff, so each atom should see at least one neighbor.
    atoms = _fake_structure("Cu", 2)
    graph = atoms_to_mace_graph(atoms, r_max=3.0)
    assert graph["edge_index"].shape[1] > 0


def test_atoms_list_to_mace_batch_concatenates_and_offsets_edges():
    atoms_list = [_fake_structure("Cu", 2), _fake_structure("Fe", 2)]
    batch = atoms_list_to_mace_batch(atoms_list, r_max=3.0)

    assert batch["n_graphs"] == 2
    assert batch["positions"].shape == (4, 3)
    assert batch["atomic_numbers"].shape == (4,)
    np.testing.assert_array_equal(batch["batch"], [0, 0, 1, 1])

    # edge_index for the second structure's atoms must be offset by the
    # first structure's atom count (2) -- no edge should ever reference an
    # index >= 4 (out of bounds) or cross between the two structures.
    assert batch["edge_index"].max() < 4
    senders, receivers = batch["edge_index"]
    sender_graph = batch["batch"][senders]
    receiver_graph = batch["batch"][receivers]
    np.testing.assert_array_equal(sender_graph, receiver_graph)


def test_atoms_list_to_mace_batch_rejects_empty_list():
    try:
        atoms_list_to_mace_batch([], r_max=3.0)
        assert False, "expected ValueError"
    except ValueError:
        pass
