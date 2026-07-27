"""
SOAP (Smooth Overlap of Atomic Positions) descriptor module.
"""

from typing import List, Optional, Union
import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP


def _normalize_atoms_distances(atoms: Atoms) -> Atoms:
    """Scales the Atoms object positions (and cell, if periodic) so that
    the minimum distance between any pair of distinct atoms is 1.0.
    """
    if len(atoms) <= 1:
        return atoms

    # Calculate all pairwise distances using Minimum Image Convention if periodic
    distances = atoms.get_all_distances(mic=True)
    
    # Fill diagonal with infinity to ignore self-distances
    np.fill_diagonal(distances, np.inf)
    
    min_dist = np.min(distances)
    if min_dist < 1e-6:
        return atoms

    scaled_atoms = atoms.copy()
    if scaled_atoms.cell and scaled_atoms.cell.any():
        scaled_atoms.set_cell(atoms.get_cell() / min_dist, scale_atoms=True)
    else:
        scaled_atoms.set_positions(atoms.get_positions() / min_dist)
        
    return scaled_atoms


def compute_soap(
    atoms: Union[Atoms, List[Atoms]],
    species: Optional[List[str]] = None,
    r_cut: float = 5.0,
    n_max: int = 8,
    l_max: int = 6,
    sigma: float = 0.5,
    periodic: Optional[bool] = None,
    crossover: bool = True,
    average: str = "off",
    sparse: bool = False,
    normalize_distances: bool = False
) -> np.ndarray:
    """Computes the SOAP descriptor for one or more ASE Atoms objects.

    Args:
        atoms: A single ASE Atoms object or a list of Atoms objects.
        species: List of chemical species to include. If None, automatically
            determined from the atoms.
        r_cut: Cutoff radius in Angstroms.
        n_max: Number of radial basis functions.
        l_max: Maximum degree of spherical harmonics.
        sigma: Standard deviation of the Gaussian positioning kernels in Angstroms.
        periodic: Whether the system is periodic. If None, determined from atoms.
        crossover: Whether to include cross-species correlations.
        average: Average type, e.g., 'off', 'inner', 'outer'.
        sparse: Whether to return a sparse matrix.
        normalize_distances: If True, normalizes the distances of each Atoms object
            such that the nearest-neighbor distance is scaled to 1.0.

    Returns:
        The SOAP descriptor representation (numpy array or sparse matrix).
    """
    if normalize_distances:
        if isinstance(atoms, Atoms):
            atoms = _normalize_atoms_distances(atoms)
        else:
            atoms = [_normalize_atoms_distances(a) for a in atoms]

    if species is None:
        if isinstance(atoms, Atoms):
            species = sorted(list(set(atoms.get_chemical_symbols())))
        else:
            all_symbols = set()
            for a in atoms:
                all_symbols.update(a.get_chemical_symbols())
            species = sorted(list(all_symbols))

    if periodic is None:
        if isinstance(atoms, Atoms):
            periodic = bool(atoms.pbc.any())
        else:
            periodic = any(bool(a.pbc.any()) for a in atoms)

    compression = {"mode": "off" if crossover else "crossover"}

    soap = SOAP(
        species=species,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        sigma=sigma,
        periodic=periodic,
        compression=compression,
        average=average,
        sparse=sparse
    )

    return soap.create(atoms)
