"""
Workflow for fetching structures, computing SOAP representation, and applying dimensionality reduction.
"""

from typing import List, Optional, Union, Tuple
import numpy as np
from ase import Atoms
from dim_red.fetch import fetch_structures_by_crystal_system
from dim_red.soap import compute_soap
from dim_red.pca import PCA
from dim_red.utils import standardize


def run_pca_reduction(
    crystal_systems: Union[str, List[str]],
    n_components: int = 2,
    api_key: Optional[str] = None,
    limit_per_system: int = 15,
    soap_kwargs: Optional[dict] = None
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Runs a complete workflow: fetches structures, computes SOAP descriptors, and reduces dimensions using PCA.

    Args:
        crystal_systems: A string or list of crystal systems to fetch.
        n_components: Number of PCA components to reduce to.
        api_key: Materials Project API key.
        limit_per_system: Maximum number of structures to fetch per crystal system.
        soap_kwargs: Additional arguments for the SOAP descriptor.

    Returns:
        X_reduced: Standardized and reduced coordinates of shape (n_samples, n_components).
        labels: A list of labels for each sample (crystal system).
        material_ids: A list of Materials Project material IDs.
    """
    if isinstance(crystal_systems, str):
        crystal_systems = [crystal_systems]

    all_atoms = []
    labels = []
    
    # Fetch structures for each crystal system
    for cs in crystal_systems:
        try:
            atoms_list = fetch_structures_by_crystal_system(
                crystal_system=cs,
                api_key=api_key,
                limit=limit_per_system
            )
            all_atoms.extend(atoms_list)
            labels.extend([cs.capitalize()] * len(atoms_list))
        except Exception as e:
            print(f"Warning: Failed to fetch structures for {cs}: {e}")

    if not all_atoms:
        raise ValueError("No structures fetched. Cannot perform dimensionality reduction.")

    # Determine unique species present across all fetched structures
    all_symbols = set()
    for atoms in all_atoms:
        all_symbols.update(atoms.get_chemical_symbols())
    species = sorted(list(all_symbols))

    # Configure SOAP parameters
    default_soap_kwargs = {
        "r_cut": 5.0,
        "n_max": 4,
        "l_max": 3,
        "average": "outer",  # average over the entire system
        "species": species
    }
    if soap_kwargs:
        default_soap_kwargs.update(soap_kwargs)

    # Compute SOAP representation for each structure
    soap_vectors = []
    material_ids = []
    for atoms in all_atoms:
        vec = compute_soap(atoms, **default_soap_kwargs)
        soap_vectors.append(vec.flatten())
        material_ids.append(atoms.info.get("material_id", "unknown"))

    X = np.array(soap_vectors)

    # Standardize the SOAP representations
    X_std = standardize(X)

    # Apply PCA reduction
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_std)

    return X_reduced, labels, material_ids
