"""
``ase.Atoms`` -> ``mace_jax`` graph-batch conversion.

Unlike ``dim_red.cgcnn.graph`` (a from-scratch reimplementation, since no
graph-NN library was available), this module wraps an external, actively
maintained JAX port of MACE (https://github.com/ACEsuit/mace-jax) rather than
reimplementing equivariant message passing -- MACE's 3-body/angular
information comes from tensor products of edge features across its
interaction layers, not from an explicit triplet/angle input the caller has
to construct, so unlike CGCNN's Gaussian-expanded *scalar* distances,
``nbr_fea`` (see ``dim_red.cgcnn.graph._gaussian_expand``), the edges built
here keep the full relative displacement vector -- MACE derives angular
correlations from that geometry itself.

IMPORTANT -- verification status: ``mace_jax`` has no public, documented
``ase.Atoms -> input dict`` conversion utility (no ASE ``Calculator``, no
``atomic_data_from_ase``-style helper as of the version checked). The field
names below (``node_attrs``, ``edge_index``, ``shifts``, ``batch``,
``positions``, ``cell``) come from reading ``mace_jax.modules.models.MACE``'s
``__call__``/``prepare_graph`` at
https://github.com/ACEsuit/mace-jax/blob/main/mace_jax/modules/models.py --
confirm these against the actually-installed ``mace_jax`` version's source
(``mace_jax/modules/models.py``'s ``prepare_graph``,
``mace_jax/tools/preprocess.py`` or similar) before relying on this in
production; adjust field names/shapes here to match if they've drifted.

Graph batching here follows jraph-style concatenation (one flat node axis and
one flat edge axis across the whole batch, with a per-node ``batch`` index
array marking which structure each atom belongs to) rather than
``dim_red.cgcnn.graph``'s fixed-``max_atoms`` padding scheme -- this is the
shape ``MACE.__call__`` consumes (its own ``scatter_sum``-based pooling reads
``data["batch"]`` to know which nodes belong to which graph), and it avoids
wasting compute on padded atoms/edges (MACE's per-edge cost is higher than
CGCNN's fixed ``max_num_nbr``, so padding to the batch's largest structure
would be comparatively more wasteful here).
"""

from typing import Any, Dict, List, Optional

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list


def atoms_to_mace_graph(atoms: Atoms, r_max: float) -> Dict[str, np.ndarray]:
    """Build one structure's MACE input graph, not yet batched with others.

    Args:
        atoms: Structure to convert.
        r_max: Cutoff radius (Angstroms) for neighbor search -- must match
            whatever cutoff the loaded pretrained checkpoint was trained
            with (``dim_red.pipeline.config.MaceConfig.r_max``).

    Returns:
        Dict with:

        - ``"positions"``: ``(n_atoms, 3)`` float32.
        - ``"atomic_numbers"``: ``(n_atoms,)`` int32 -- real atomic numbers
          (unlike ``dim_red.cgcnn.graph``'s per-structure local species
          slots: MACE's pretrained embedding table is keyed by real chemical
          species, since it was trained across many elements at once).
        - ``"edge_index"``: ``(2, n_edges)`` int32, ``[senders, receivers]``.
        - ``"shifts"``: ``(n_edges, 3)`` float32 -- real-space periodic-image
          shift vectors (``cell_shift @ cell``), so
          ``positions[receivers] + shifts - positions[senders]`` is the true
          minimum-image-aware displacement even across a periodic boundary.
        - ``"cell"``: ``(3, 3)`` float32.
    """
    positions = np.asarray(atoms.get_positions(), dtype=np.float32)
    atomic_numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int32)
    # np.array(...), not np.asarray(..., dtype=...) -- ase.cell.Cell's
    # __array__(dtype=None, copy=False) refuses an implicit dtype-casting
    # copy under numpy>=2.0 when called that way (same reason
    # dim_red.augmentation reads a cell as `np.array(atoms.get_cell())`).
    cell = np.array(atoms.get_cell()).astype(np.float32)

    senders, receivers, cell_shifts = neighbor_list("ijS", atoms, r_max)
    shifts = (np.asarray(cell_shifts, dtype=np.float32) @ cell).astype(np.float32)
    edge_index = np.stack(
        [np.asarray(senders, dtype=np.int32), np.asarray(receivers, dtype=np.int32)],
        axis=0,
    )

    return {
        "positions": positions,
        "atomic_numbers": atomic_numbers,
        "edge_index": edge_index,
        "shifts": shifts,
        "cell": cell,
    }


def atoms_list_to_mace_batch(
    atoms_list: List[Atoms], r_max: float
) -> Dict[str, np.ndarray]:
    """Build one jraph-style batched MACE graph from a whole list of structures.

    Every structure's nodes/edges are concatenated along a single flat axis;
    ``edge_index`` values are offset by each structure's cumulative atom
    count so they keep indexing correctly into the concatenated
    ``positions``/``atomic_numbers`` arrays, and ``batch`` records which
    structure each node (row) belongs to -- ``mace_jax``'s own pooling
    (``dim_red.mace.model``, via ``scatter_sum``/``scatter_mean`` over
    ``batch``) uses this to know which nodes to pool together.

    Args:
        atoms_list: Structures to batch (non-empty).
        r_max: Cutoff radius (Angstroms) -- see ``atoms_to_mace_graph``.

    Returns:
        Dict with ``"positions"`` ``(total_atoms, 3)``, ``"atomic_numbers"``
        ``(total_atoms,)``, ``"edge_index"`` ``(2, total_edges)``,
        ``"shifts"`` ``(total_edges, 3)``, ``"batch"`` ``(total_atoms,)`` int32
        (structure index per atom), and ``"n_graphs"`` (int, the number of
        structures batched -- ``len(atoms_list)``).

    Raises:
        ValueError: If ``atoms_list`` is empty.
    """
    if not atoms_list:
        raise ValueError("atoms_list must not be empty")

    positions_parts = []
    atomic_numbers_parts = []
    edge_index_parts = []
    shifts_parts = []
    batch_parts = []

    atom_offset = 0
    for graph_idx, atoms in enumerate(atoms_list):
        graph = atoms_to_mace_graph(atoms, r_max)
        n_atoms = graph["positions"].shape[0]

        positions_parts.append(graph["positions"])
        atomic_numbers_parts.append(graph["atomic_numbers"])
        edge_index_parts.append(graph["edge_index"] + atom_offset)
        shifts_parts.append(graph["shifts"])
        batch_parts.append(np.full(n_atoms, graph_idx, dtype=np.int32))

        atom_offset += n_atoms

    return {
        "positions": np.concatenate(positions_parts, axis=0),
        "atomic_numbers": np.concatenate(atomic_numbers_parts, axis=0),
        "edge_index": np.concatenate(edge_index_parts, axis=1),
        "shifts": np.concatenate(shifts_parts, axis=0),
        "batch": np.concatenate(batch_parts, axis=0),
        "n_graphs": len(atoms_list),
    }
