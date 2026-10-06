"""
SOAP (Smooth Overlap of Atomic Positions) descriptor module.
"""

from typing import List, Optional, Union

import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP


def compute_soap(
    atoms: Union[Atoms, List[Atoms]],
    species: Optional[List[str]] = None,
    r_cut: float = 5.0,
    n_max: int = 8,
    l_max: int = 6,
    sigma: float = 0.5,
    periodic: Optional[bool] = None,
    element_agnostic: bool = False,
    average: str = "off",
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
        element_agnostic: If True, compresses the descriptor into an
            element-agnostic representation (dscribe's "mu2" compression
            mode), so its size no longer depends on the number of species.
        average: Average type, e.g., 'off', 'inner', 'outer'. Use 'outer' to
            get a single global descriptor per structure, computed as the
            average of the per-atom SOAP vectors.

    Returns:
        The SOAP descriptor representation (numpy array).
    """
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

    compression = {"mode": "mu2" if element_agnostic else "off"}

    soap = SOAP(
        species=species,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        sigma=sigma,
        periodic=periodic,
        compression=compression,
        average=average,
    )

    return soap.create(atoms)
