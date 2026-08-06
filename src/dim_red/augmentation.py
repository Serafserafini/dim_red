"""
Data augmentation for diversifying a database of crystal structures.

Meant to sit immediately after ``dim_red.fetch.fetch_structures_by_crystal_system``:
takes the fetched ``Atoms`` objects and produces additional, slightly perturbed
copies of each one, either by applying a small random positional jitter to the
atoms or by randomly deleting atoms (vacancies). Both mechanisms are
independently configurable (probability of being applied, and the
magnitude/probability of the perturbation itself) via ``AugmentationConfig``,
but every augmented copy is guaranteed to have at least one of the two
actually applied to it -- see ``_augment_once``.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from ase import Atoms

from dim_red.fetch import fetch_structures_by_crystal_system


@dataclass
class AugmentationConfig:
    """Configuration for structure augmentation.

    Every augmented copy is guaranteed to have at least one of jitter/vacancy
    actually applied to it (see ``_augment_once``) -- a copy where neither
    mechanism's probability roll succeeds still gets one of them forced on,
    rather than coming out identical to the original. A corollary: if only
    one of the two mechanisms is configured at all (``jitter_std > 0`` xor
    ``vacancy_atom_probability > 0``), that one applies to *every* augmented
    copy regardless of its own probability, since it's the only thing that
    can be forced.

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
        max_vacancies: Maximum number of atoms removed per augmented copy
            when vacancy removal applies. ``None`` (default) leaves it
            uncapped, short of the structure's own floor of 1 remaining atom
            (removal never drops a structure to 0 atoms, regardless of this
            setting).
        supercell_radius: If set (Angstroms), every structure is first
            expanded into a supercell (see ``make_supercell_for_radius``)
            large enough to fit a sphere of this radius before any
            jitter/vacancy is applied -- meant for unit cells too small to
            meaningfully augment (e.g. pyxtal can generate cells holding as
            few as 1-2 atoms). ``None`` (default) skips this step entirely,
            current behavior unchanged. Applies to the kept-original copy
            too, not just augmented ones, since the point is giving *every*
            output structure enough atoms to work with.
        seed: Optional seed for reproducible augmentation.
    """

    n_augmented: int = 1
    keep_original: bool = True
    jitter_probability: float = 0.5
    jitter_std: float = 0.05
    vacancy_probability: float = 0.0
    vacancy_atom_probability: float = 0.05
    max_vacancies: Optional[int] = None
    supercell_radius: Optional[float] = None
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
        if self.max_vacancies is not None and self.max_vacancies < 0:
            raise ValueError("max_vacancies must be >= 0")
        if self.supercell_radius is not None and self.supercell_radius <= 0:
            raise ValueError("supercell_radius must be > 0")


def make_supercell_for_radius(atoms: Atoms, radius: float) -> Atoms:
    """Returns a supercell of ``atoms``, repeated an integer number of times
    along each periodic lattice vector, large enough that a sphere of
    ``radius`` Angstroms fits inside without touching its own periodic
    image in any direction.

    Uses the standard "perpendicular cell width >= 2 * radius" rule for
    sizing a simulation cell to a given interaction cutoff: for lattice
    vectors ``a``/``b``/``c``, the perpendicular width in the direction of
    ``a`` is ``volume / |b x c|`` (the height of the cell's parallelepiped
    along that axis), and repeating ``n`` times along ``a`` scales that
    width by exactly ``n`` (the other two vectors, and hence ``|b x c|``,
    are untouched) -- so each axis's required repeat count can be computed
    independently from the original, unrepeated cell.

    Non-periodic axes (``atoms.pbc[i]`` is ``False``) are never repeated.
    Always returns a new ``Atoms`` object (a plain copy when no axis needs
    repeating), preserving ``atoms.info``.
    """
    cell = np.array(atoms.get_cell())
    volume = abs(np.linalg.det(cell))

    repeats = []
    for i in range(3):
        if not atoms.pbc[i] or volume == 0:
            repeats.append(1)
            continue
        other = [cell[j] for j in range(3) if j != i]
        cross_norm = np.linalg.norm(np.cross(other[0], other[1]))
        perpendicular_width = volume / cross_norm if cross_norm > 0 else 0.0
        if perpendicular_width <= 0:
            repeats.append(1)
        else:
            # A small tolerance absorbs floating-point noise in the
            # volume/cross-product computation above so an exact-fit ratio
            # (e.g. 5.000000000000001 that should be 5) doesn't spuriously
            # round up to one extra, unneeded repeat.
            ratio = 2 * radius / perpendicular_width
            repeats.append(max(1, int(np.ceil(ratio - 1e-9))))

    return atoms.repeat(tuple(repeats))


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
    max_vacancies: Optional[int] = None,
) -> Atoms:
    """Returns a copy of ``atoms`` with each atom independently removed with
    probability ``atom_probability`` (a vacancy defect), capped at
    ``max_vacancies`` total removals (``None`` -> uncapped) -- and, either
    way, never dropping the structure below 1 remaining atom.
    """
    n = len(atoms)
    if n <= 1 or atom_probability <= 0:
        return atoms.copy()

    remove_indices = np.flatnonzero(rng.random(n) < atom_probability)
    hard_cap = n - 1  # always keep at least one atom
    effective_cap = hard_cap if max_vacancies is None else min(max_vacancies, hard_cap)
    if remove_indices.size > effective_cap:
        remove_indices = rng.choice(remove_indices, size=effective_cap, replace=False)

    keep_mask = np.ones(n, dtype=bool)
    keep_mask[remove_indices] = False
    return atoms[np.flatnonzero(keep_mask)]


def _augment_once(
    atoms: Atoms, config: AugmentationConfig, rng: np.random.Generator
) -> Atoms:
    """Generates a single augmented copy, independently sampling whether
    jitter and/or vacancy removal apply, per ``config``'s probabilities --
    then, if neither roll succeeded, forces one of them on so every
    augmented copy differs from the original by at least one mechanism (see
    ``AugmentationConfig``'s docstring for the corollary this has when only
    one mechanism is configured at all).
    """
    jitter_available = config.jitter_std > 0
    vacancy_available = config.vacancy_atom_probability > 0

    apply_jitter = jitter_available and rng.random() < config.jitter_probability
    apply_vacancy = vacancy_available and rng.random() < config.vacancy_probability

    if not apply_jitter and not apply_vacancy:
        if jitter_available and vacancy_available:
            if rng.random() < 0.5:
                apply_jitter = True
            else:
                apply_vacancy = True
        elif jitter_available:
            apply_jitter = True
        elif vacancy_available:
            apply_vacancy = True
        # else: neither mechanism is configured at all -- nothing to force.

    augmented = atoms
    applied: List[str] = []

    if apply_jitter:
        augmented = jitter_positions(augmented, config.jitter_std, rng)
        applied.append("jitter")

    if apply_vacancy:
        augmented = remove_random_atoms(
            augmented, config.vacancy_atom_probability, rng, config.max_vacancies
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

    For every structure in ``atoms_list``, first expands it into a supercell
    when ``config.supercell_radius`` is set (see ``make_supercell_for_radius``
    -- done once per input structure, before anything else, so both the
    kept-original copy and every augmented copy are built from it), then
    generates ``config.n_augmented`` augmented copies (each independently
    jittered and/or given vacancies according to ``config``'s probabilities),
    optionally keeping the (possibly supercell-expanded) original alongside
    them.

    Returns:
        The diversified list of ``Atoms`` objects. Each has
        ``info["augmented"]`` (bool), ``info["augmentations_applied"]``
        (list of ``"jitter"``/``"vacancy"``) and ``info["source_material_id"]``
        set, in addition to whatever ``info`` it already carried.
    """
    rng = np.random.default_rng(config.seed)
    result: List[Atoms] = []
    for atoms in atoms_list:
        base = (
            make_supercell_for_radius(atoms, config.supercell_radius)
            if config.supercell_radius is not None
            else atoms
        )
        if config.keep_original:
            original = base.copy()
            original.info["augmented"] = False
            original.info["augmentations_applied"] = []
            original.info["source_material_id"] = atoms.info.get("material_id")
            result.append(original)
        for _ in range(config.n_augmented):
            result.append(_augment_once(base, config, rng))
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
