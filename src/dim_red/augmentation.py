"""
Data augmentation for diversifying a database of crystal structures.

Meant to sit immediately after ``dim_red.fetch.fetch_structures_by_crystal_system``:
takes the fetched ``Atoms`` objects and produces additional, slightly perturbed
copies of each one, either by applying a small random positional jitter to the
atoms or by randomly deleting atoms (vacancies). Both mechanisms are
independently configurable (probability of being applied, and the
magnitude/probability of the perturbation itself) via ``AugmentationConfig``.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from ase import Atoms

from dim_red.fetch import fetch_structures_by_crystal_system


@dataclass
class AugmentationConfig:
    """Configuration for structure augmentation.

    Attributes:
        n_augmented: Number of augmented copies generated per input structure.
        keep_original: Whether the unmodified structure is also included in
            the output, alongside its augmented copies.
        jitter_probability: Probability, per augmented copy, that positional
            jitter is applied to it at all.
        jitter_std: Standard deviation (in Angstroms) of the Gaussian noise
            added to atomic positions when jitter is applied.
        vacancy_probability: Probability, per augmented copy, that vacancy
            removal is applied to it at all.
        vacancy_atom_probability: Probability that any individual atom is
            removed, once a copy has been selected for vacancy removal.
        min_atoms: Minimum number of atoms a structure must retain; vacancy
            removal restores randomly chosen atoms if it would otherwise drop
            below this floor.
        seed: Optional seed for reproducible augmentation.
    """

    n_augmented: int = 1
    keep_original: bool = True
    jitter_probability: float = 0.5
    jitter_std: float = 0.05
    vacancy_probability: float = 0.0
    vacancy_atom_probability: float = 0.05
    min_atoms: int = 1
    seed: Optional[int] = None

    def __post_init__(self):
        if self.n_augmented < 0:
            raise ValueError("n_augmented must be >= 0")
        if not 0.0 <= self.jitter_probability <= 1.0:
            raise ValueError("jitter_probability must be in [0, 1]")
        if not 0.0 <= self.vacancy_probability <= 1.0:
            raise ValueError("vacancy_probability must be in [0, 1]")
        if not 0.0 <= self.vacancy_atom_probability <= 1.0:
            raise ValueError("vacancy_atom_probability must be in [0, 1]")
        if self.jitter_std < 0:
            raise ValueError("jitter_std must be >= 0")
        if self.min_atoms < 1:
            raise ValueError("min_atoms must be >= 1")


def jitter_positions(atoms: Atoms, std: float, rng: np.random.Generator) -> Atoms:
    """Returns a copy of ``atoms`` with isotropic Gaussian noise (std in
    Angstroms) added to every atomic position.
    """
    jittered = atoms.copy()
    if std <= 0:
        return jittered
    noise = rng.normal(loc=0.0, scale=std, size=jittered.positions.shape)
    jittered.set_positions(jittered.get_positions() + noise)
    return jittered


def remove_random_atoms(
    atoms: Atoms,
    atom_probability: float,
    rng: np.random.Generator,
    min_atoms: int = 1,
) -> Atoms:
    """Returns a copy of ``atoms`` with each atom independently removed with
    probability ``atom_probability`` (a vacancy defect), restoring randomly
    chosen atoms if removal would otherwise drop the count below ``min_atoms``.
    """
    n = len(atoms)
    if n <= min_atoms or atom_probability <= 0:
        return atoms.copy()

    remove_mask = rng.random(n) < atom_probability
    keep_indices = np.flatnonzero(~remove_mask)
    if keep_indices.size < min_atoms:
        removed_indices = np.flatnonzero(remove_mask)
        n_restore = min_atoms - keep_indices.size
        restored = rng.choice(removed_indices, size=n_restore, replace=False)
        keep_indices = np.sort(np.concatenate([keep_indices, restored]))
    return atoms[keep_indices]


def _augment_once(
    atoms: Atoms, config: AugmentationConfig, rng: np.random.Generator
) -> Atoms:
    """Generates a single augmented copy, independently sampling whether
    jitter and/or vacancy removal apply, per ``config``'s probabilities.
    """
    augmented = atoms
    applied: List[str] = []

    if config.jitter_std > 0 and rng.random() < config.jitter_probability:
        augmented = jitter_positions(augmented, config.jitter_std, rng)
        applied.append("jitter")

    if (
        config.vacancy_atom_probability > 0
        and rng.random() < config.vacancy_probability
    ):
        augmented = remove_random_atoms(
            augmented, config.vacancy_atom_probability, rng, config.min_atoms
        )
        applied.append("vacancy")

    if not applied:
        augmented = augmented.copy()

    augmented.info["augmented"] = bool(applied)
    augmented.info["augmentations_applied"] = applied
    augmented.info["source_material_id"] = atoms.info.get("material_id")
    return augmented


def augment_structures(
    atoms_list: Sequence[Atoms], config: AugmentationConfig
) -> List[Atoms]:
    """Builds a diversified database of structures via data augmentation.

    For every structure in ``atoms_list``, generates ``config.n_augmented``
    augmented copies (each independently jittered and/or given vacancies
    according to ``config``'s probabilities), optionally keeping the original
    structure alongside them.

    Returns:
        The diversified list of ``Atoms`` objects. Each has
        ``info["augmented"]`` (bool), ``info["augmentations_applied"]``
        (list of ``"jitter"``/``"vacancy"``) and ``info["source_material_id"]``
        set, in addition to whatever ``info`` it already carried.
    """
    rng = np.random.default_rng(config.seed)
    result: List[Atoms] = []
    for atoms in atoms_list:
        if config.keep_original:
            original = atoms.copy()
            original.info["augmented"] = False
            original.info["augmentations_applied"] = []
            original.info["source_material_id"] = atoms.info.get("material_id")
            result.append(original)
        for _ in range(config.n_augmented):
            result.append(_augment_once(atoms, config, rng))
    return result


def fetch_and_augment_structures(
    crystal_system,
    augmentation_config: AugmentationConfig,
    api_key: Optional[str] = None,
    limit: int = 10,
) -> List[Atoms]:
    """Fetches structures from Materials Project and immediately augments
    them, i.e. ``augment_structures`` chained right after
    ``fetch_structures_by_crystal_system``.
    """
    atoms_list = fetch_structures_by_crystal_system(
        crystal_system=crystal_system, api_key=api_key, limit=limit
    )
    return augment_structures(atoms_list, augmentation_config)
