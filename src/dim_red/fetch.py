"""
Module for fetching crystal structures from the Materials Project database.
"""

import os
from typing import List, Optional, Union

from ase import Atoms
from mp_api.client import MPRester
from pymatgen.io.ase import AseAtomsAdaptor


def fetch_structures_by_crystal_system(
    crystal_system: Union[str, List[str]],
    api_key: Optional[str] = None,
    limit: int = 10,
) -> List[Atoms]:
    """Fetches crystal structures from the Materials Project based on the crystal system.

    Args:
        crystal_system: A string or list of strings representing the crystal system(s)
            (e.g., 'cubic', 'orthorhombic', 'hexagonal', 'tetragonal', 'monoclinic',
            'triclinic', 'trigonal').
        api_key: Materials Project API key. If None, it will look for the MP_API_KEY
            environment variable.
        limit: The maximum number of structures to retrieve.

    Returns:
        A list of ASE Atoms objects.

    Raises:
        ValueError: If no API key is provided and MP_API_KEY environment variable is not set.
    """
    api_key = api_key or os.environ.get("MP_API_KEY")
    if not api_key:
        raise ValueError(
            "Materials Project API key is required. "
            "Please provide it as an argument or set the 'MP_API_KEY' environment variable."
        )

    # Normalize crystal system input (e.g. capitalize)
    if isinstance(crystal_system, str):
        crystal_systems = [crystal_system.capitalize()]
    else:
        crystal_systems = [cs.capitalize() for cs in crystal_system]

    # Query Materials Project
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            crystal_system=crystal_systems,
            fields=["material_id", "structure", "symmetry"],
            chunk_size=limit,
            num_chunks=1,
        )

    # Ensure we return at most 'limit' structures
    docs = docs[:limit]

    # Convert pymatgen structures to ASE Atoms
    atoms_list = []
    for doc in docs:
        if hasattr(doc, "structure") and doc.structure is not None:
            atoms = AseAtomsAdaptor.get_atoms(doc.structure)
            # Add material_id as an info attribute for reference
            atoms.info["material_id"] = str(doc.material_id)
            # Add the spacegroup number (1-230) as an info attribute, when available
            symmetry = getattr(doc, "symmetry", None)
            if symmetry is not None and getattr(symmetry, "number", None) is not None:
                atoms.info["spacegroup"] = int(symmetry.number)
            atoms_list.append(atoms)

    return atoms_list
