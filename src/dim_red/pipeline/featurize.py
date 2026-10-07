"""Features of *new* structures, computed exactly like training did: SOAP with
the ``mu2`` (element-agnostic) compression -- which depends only on geometry,
so the species list never has to match any training dataset -- or MACE
embeddings, then the per-stack standardization saved with each stack."""

from typing import List

import numpy as np
from ase import Atoms

from dim_red.pipeline.full_stack_config import StackDataConfig
from dim_red.soap import compute_soap
from dim_red.utils import apply_standardization


def featurize_structures(
    atoms_list: List[Atoms], model_kind: str, data: StackDataConfig
) -> np.ndarray:
    """Raw (not yet standardized) global feature vector per structure.

    Periodicity is decided by ``compute_soap`` over the whole list (as in
    training): keep periodic and non-periodic structures in separate calls."""
    if not atoms_list:
        raise ValueError("atoms_list is empty -- nothing to featurize")
    if model_kind == "supcon_mace":
        from dim_red.mace.model import MaceEncoder

        encoder = MaceEncoder(**data.mace.mace_kwargs())
        return np.asarray(encoder.encode(atoms_list), dtype=np.float32)

    kwargs = data.soap.as_kwargs()
    if not kwargs["element_agnostic"]:
        raise ValueError(
            "featurizing new structures needs data.soap.element_agnostic: true "
            "(SOAP 'mu2'): without it the feature size depends on the training "
            "dataset's species"
        )
    present = {s for a in atoms_list for s in a.get_chemical_symbols()}
    kwargs["species"] = sorted(present | set(kwargs.get("species") or []))
    kwargs["average"] = "outer"
    vectors = compute_soap(atoms_list, **kwargs)
    return np.asarray(vectors, dtype=np.float32).reshape(len(atoms_list), -1)


def standardize(raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return apply_standardization(raw, mean, std).astype(np.float32)
