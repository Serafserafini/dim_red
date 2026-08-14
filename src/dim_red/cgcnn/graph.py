"""
Crystal-graph construction for CGCNN (Xie & Grossman 2018), reimplemented
from scratch against this codebase's own conventions rather than the
original repo's PyTorch/pymatgen data pipeline -- pure NumPy/ASE, no jax
dependency at all (jax only enters at ``dim_red.cgcnn.model``).

Two deliberate deviations from the original paper, both explained in
``CLAUDE.md``:

- Neighbor search uses ``ase.neighborlist.neighbor_list`` rather than
  pymatgen's ``Structure.get_all_neighbors`` or a naive minimum-image-
  convention distance matrix -- the latter is only valid for cutoffs below
  half the cell's shortest perpendicular width, routinely violated by
  CGCNN's own default ``radius=8.0`` against small pyxtal-generated cells.
- Atom features are keyed by a per-structure LOCAL species slot (see
  :func:`_local_species_indices`), not a real atomic number or the original
  repo's fixed ``atom_init.json`` lookup table -- this model must be able to
  distinguish species A from species B within a structure without ever
  caring which real chemical element either one is.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list


def _neighbors_for_structure(
    atoms: Atoms, radius: float, max_num_nbr: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-atom neighbor lists, capped at ``max_num_nbr`` closest neighbors.

    Uses ``ase.neighborlist.neighbor_list('ijd', atoms, radius)``, which
    handles periodic images via cell-shift search (correct even when
    ``radius`` exceeds half the cell's perpendicular width, unlike a naive
    minimum-image-convention distance matrix) and works unchanged for
    non-periodic ``Atoms`` too.

    Args:
        atoms: Structure to search.
        radius: Cutoff radius (Angstroms).
        max_num_nbr: Maximum neighbors kept per atom -- the closest ones
            within ``radius``, sorted ascending by distance.

    Returns:
        ``(nbr_idx, nbr_dist, nbr_mask)``, each shape
        ``(len(atoms), max_num_nbr)``. Atoms with fewer than
        ``max_num_nbr`` neighbors within ``radius`` are padded: ``nbr_idx``
        with ``0`` (this structure's atom 0), ``nbr_dist`` with ``radius``,
        ``nbr_mask`` with ``0.0`` -- downstream consumers must always
        multiply by ``nbr_mask`` rather than assume unmasked slots are
        meaningless zeros.
    """
    n_atoms = len(atoms)
    nbr_idx = np.zeros((n_atoms, max_num_nbr), dtype=np.int32)
    nbr_dist = np.full((n_atoms, max_num_nbr), radius, dtype=np.float32)
    nbr_mask = np.zeros((n_atoms, max_num_nbr), dtype=np.float32)

    if n_atoms == 0:
        return nbr_idx, nbr_dist, nbr_mask

    i, j, d = neighbor_list("ijd", atoms, radius)
    for atom in range(n_atoms):
        row_mask = i == atom
        js = j[row_mask]
        ds = d[row_mask]
        order = np.argsort(ds)[:max_num_nbr]
        k = len(order)
        nbr_idx[atom, :k] = js[order]
        nbr_dist[atom, :k] = ds[order]
        nbr_mask[atom, :k] = 1.0

    return nbr_idx, nbr_dist, nbr_mask


def _gaussian_expand(
    distances: np.ndarray, dmin: float, dmax: float, step: float
) -> np.ndarray:
    """Expand scalar distances into a Gaussian radial basis.

    ``exp(-((d - filter_k)^2) / step^2)`` for ``filter_k`` spanning
    ``[dmin, dmax]`` in ``n_gaussian`` evenly-spaced steps (``np.linspace``,
    not ``np.arange``, to avoid float step-accumulation off-by-ones in the
    filter count).

    Args:
        distances: Array of any shape.
        dmin: Lower bound of the filter bank.
        dmax: Upper bound of the filter bank.
        step: Spacing between filters.

    Returns:
        Array of shape ``distances.shape + (n_gaussian,)``, where
        ``n_gaussian = round((dmax - dmin) / step) + 1``.
    """
    n_gaussian = int(round((dmax - dmin) / step)) + 1
    filters = np.linspace(dmin, dmax, n_gaussian, dtype=np.float32)
    diff = distances[..., None].astype(np.float32) - filters
    return np.exp(-(diff**2) / (step**2)).astype(np.float32)


def _local_species_indices(atomic_numbers: np.ndarray, max_species: int) -> np.ndarray:
    """Remap real atomic numbers to a per-structure LOCAL species slot.

    Distinct atomic numbers present in this one structure are sorted
    ascending and assigned slots ``1..k`` (``0`` reserved for padding, so it
    never collides with a real slot). A binary Fe/O structure and a binary
    Na/Cl structure with the same geometry produce identical output --
    intentional: the model must distinguish species A from species B within
    a structure, never which real chemical element either one is.

    Args:
        atomic_numbers: Real atomic numbers, shape ``(n_atoms,)``.
        max_species: Maximum distinct species this structure may have.

    Returns:
        Local species slot per atom, shape ``(n_atoms,)``, values ``1..k``
        where ``k <= max_species`` is the number of distinct species
        actually present.

    Raises:
        ValueError: If more than ``max_species`` distinct species are present.
    """
    distinct = np.unique(atomic_numbers)
    if len(distinct) > max_species:
        raise ValueError(
            f"Structure has {len(distinct)} distinct species, exceeding "
            f"max_species={max_species} -- raise graph.max_species or "
            "reduce the structure's species count."
        )
    remap = {int(z): slot + 1 for slot, z in enumerate(distinct)}
    return np.array([remap[int(z)] for z in atomic_numbers], dtype=np.int32)


def atoms_to_graph(
    atoms: Atoms,
    radius: float = 8.0,
    max_num_nbr: int = 12,
    dmin: float = 0.0,
    dmax: Optional[float] = None,
    step: float = 0.2,
    max_species: int = 10,
) -> Dict[str, np.ndarray]:
    """Build one structure's graph, not yet padded to a dataset-wide ``max_atoms``.

    Args:
        atoms: Structure to convert.
        radius: Cutoff radius (Angstroms) for neighbor search.
        max_num_nbr: Maximum neighbors kept per atom.
        dmin: Lower bound of the Gaussian distance-expansion filter bank.
        dmax: Upper bound of the filter bank. ``None`` (default) resolves
            to ``radius``.
        step: Spacing between Gaussian filters.
        max_species: Maximum distinct species this structure may have --
            see :func:`_local_species_indices`.

    Returns:
        Dict with ``"local_species_idx"`` (``(n_atoms,)`` int32),
        ``"nbr_idx"``/``"nbr_mask"`` (``(n_atoms, max_num_nbr)``), and
        ``"nbr_fea"`` (``(n_atoms, max_num_nbr, n_gaussian)`` float32).

    Raises:
        ValueError: If the structure has more than ``max_species`` distinct species.
    """
    resolved_dmax = dmax if dmax is not None else radius
    atomic_numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
    local_species_idx = _local_species_indices(atomic_numbers, max_species)
    nbr_idx, nbr_dist, nbr_mask = _neighbors_for_structure(atoms, radius, max_num_nbr)
    nbr_fea = _gaussian_expand(nbr_dist, dmin, resolved_dmax, step)
    return {
        "local_species_idx": local_species_idx,
        "nbr_idx": nbr_idx,
        "nbr_fea": nbr_fea,
        "nbr_mask": nbr_mask,
    }


def atoms_list_to_graph_arrays(
    atoms_list: List[Atoms],
    radius: float = 8.0,
    max_num_nbr: int = 12,
    dmin: float = 0.0,
    dmax: Optional[float] = None,
    step: float = 0.2,
    max_species: int = 10,
    max_atoms: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build whole-dataset graph arrays, padded to a common ``max_atoms``.

    ``max_atoms`` is purely an array-shape convenience for this one call --
    it is not baked into any trained parameter (every downstream ``Dense``/
    ``Embed`` layer only ever sees the trailing feature axis), so a later
    call against different structures (e.g. encoding brand-new structures
    for inference) is free to resolve its own, different ``max_atoms`` with
    no compatibility requirement against whatever ``max_atoms`` training used.

    Args:
        atoms_list: Structures to convert (non-empty).
        radius: Cutoff radius (Angstroms) for neighbor search.
        max_num_nbr: Maximum neighbors kept per atom.
        dmin: Lower bound of the Gaussian distance-expansion filter bank.
        dmax: Upper bound of the filter bank. ``None`` (default) resolves to ``radius``.
        step: Spacing between Gaussian filters.
        max_species: Maximum distinct species any one structure may have.
        max_atoms: Pad every structure's atom axis to this length. ``None``
            (default) resolves to the largest structure's own atom count.

    Returns:
        ``(local_species_idx, nbr_idx, nbr_fea, nbr_mask, atom_mask)``:

        - ``local_species_idx``: ``(n, max_atoms)`` int32.
        - ``nbr_idx``: ``(n, max_atoms, max_num_nbr)`` int32.
        - ``nbr_fea``: ``(n, max_atoms, max_num_nbr, n_gaussian)`` float32.
        - ``nbr_mask``: ``(n, max_atoms, max_num_nbr)`` float32.
        - ``atom_mask``: ``(n, max_atoms)`` float32 -- ``1.0`` for real
          atoms, ``0.0`` for padding rows.

    Raises:
        ValueError: If ``atoms_list`` is empty, if an explicit ``max_atoms``
            is smaller than some structure's actual atom count, or if any
            structure has more than ``max_species`` distinct species.
    """
    if not atoms_list:
        raise ValueError("atoms_list must not be empty")

    atom_counts = [len(a) for a in atoms_list]
    largest = max(atom_counts)
    if max_atoms is not None and max_atoms < largest:
        raise ValueError(
            f"max_atoms={max_atoms} is smaller than the largest structure's "
            f"atom count ({largest})"
        )
    resolved_max_atoms = max_atoms if max_atoms is not None else largest

    resolved_dmax = dmax if dmax is not None else radius
    n_gaussian = int(round((resolved_dmax - dmin) / step)) + 1
    n = len(atoms_list)

    local_species_idx = np.zeros((n, resolved_max_atoms), dtype=np.int32)
    nbr_idx = np.zeros((n, resolved_max_atoms, max_num_nbr), dtype=np.int32)
    nbr_fea = np.zeros(
        (n, resolved_max_atoms, max_num_nbr, n_gaussian), dtype=np.float32
    )
    nbr_mask = np.zeros((n, resolved_max_atoms, max_num_nbr), dtype=np.float32)
    atom_mask = np.zeros((n, resolved_max_atoms), dtype=np.float32)

    for idx, atoms in enumerate(atoms_list):
        graph = atoms_to_graph(
            atoms,
            radius=radius,
            max_num_nbr=max_num_nbr,
            dmin=dmin,
            dmax=dmax,
            step=step,
            max_species=max_species,
        )
        n_atoms = atom_counts[idx]
        local_species_idx[idx, :n_atoms] = graph["local_species_idx"]
        nbr_idx[idx, :n_atoms] = graph["nbr_idx"]
        nbr_fea[idx, :n_atoms] = graph["nbr_fea"]
        nbr_mask[idx, :n_atoms] = graph["nbr_mask"]
        atom_mask[idx, :n_atoms] = 1.0

    return local_species_idx, nbr_idx, nbr_fea, nbr_mask, atom_mask
