"""
Alternative to fetching structures from Materials Project (``dim_red.fetch``):
builds a synthetic crystal-structure database from scratch with ``pyxtal``,
generating random-but-symmetric structures per space group instead of
querying an external API. Not a dim_red dependency by default -- requires
``pip install pyxtal`` (see ``CLAUDE.md``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from ase import Atoms
from pyxtal import pyxtal
from pyxtal.msg import Comp_CompatibilityError
from pyxtal.symmetry import Group

logger = logging.getLogger("dim_red.generate")

_ALL_FAMILIES = [
    "Triclinic",
    "Monoclinic",
    "Orthorhombic",
    "Tetragonal",
    "Trigonal",
    "Hexagonal",
    "Cubic",
]

# Atom-count-per-species values retried in order until one is compatible with
# a space group's Wyckoff multiplicities (see GenerationConfig.candidate_num_ions).
_DEFAULT_CANDIDATE_NUM_IONS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 48, 96, 192]

# Chemically unremarkable elements (spread across valences/covalent radii),
# used as placeholder species when the caller only cares about *how many*
# distinct species a structure has, not which ones.
_DEFAULT_SPECIES_POOL = [
    "C",
    "Si",
    "Ge",
    "Sn",
    "N",
    "P",
    "As",
    "O",
    "S",
    "Se",
    "Al",
    "Ga",
    "In",
    "Zn",
    "Cd",
    "Fe",
    "Ti",
    "Cu",
    "Mg",
    "Ca",
]


@dataclass
class GenerationConfig:
    """Configuration for pyxtal-based synthetic structure generation.

    Exactly one of ``structures_per_spacegroup``/``structures_per_family``
    must be set; ``families``/``spacegroups`` are mutually exclusive ways to
    restrict which space groups are in scope (default: all 230).

    Attributes:
        families: Crystal families to generate for (e.g.
            ``["Cubic", "Hexagonal"]``); ``None`` generates across all 7.
            Mutually exclusive with ``spacegroups``.
        spacegroups: Explicit space-group numbers (1-230) to generate for,
            bypassing family enumeration. Mutually exclusive with ``families``.
        structures_per_spacegroup: If set, generate exactly this many
            structures for *every* space group in scope.
        structures_per_family: If set, generate this many structures in
            total *per family*, split across that family's space groups (in
            scope) according to ``distribution``.
        distribution: How ``structures_per_family`` is split across a
            family's space groups -- ``"uniform"`` (as evenly as possible) or
            ``"random"`` (each of the family's requested structures assigned
            to a uniformly random space group in that family). Ignored when
            ``structures_per_spacegroup`` is used instead.
        n_species: Number of distinct chemical species per generated
            structure. Which elements are used doesn't matter -- ``n_species``
            of them are sampled (without replacement) from ``species_pool``
            independently for each structure.
        species_pool: Element symbols to sample species from. Defaults to a
            built-in pool of chemically unremarkable elements.
        candidate_num_ions: Atom-count-per-species values tried in order
            until one is compatible with a space group's Wyckoff
            multiplicities.
        factor: Volume factor passed to ``pyxtal.pyxtal.from_random``.
        max_count: ``pyxtal``'s own internal retry budget per attempt.
        seed: Optional seed for reproducible generation (species selection,
            the "random" distribution mode, and pyxtal's own RNG).
    """

    families: Optional[List[str]] = None
    spacegroups: Optional[List[int]] = None
    structures_per_spacegroup: Optional[int] = None
    structures_per_family: Optional[int] = None
    distribution: str = "uniform"
    n_species: int = 1
    species_pool: Optional[List[str]] = None
    candidate_num_ions: List[int] = field(
        default_factory=lambda: list(_DEFAULT_CANDIDATE_NUM_IONS)
    )
    factor: float = 1.1
    max_count: int = 5
    seed: Optional[int] = None

    def __post_init__(self):
        if self.families is not None and self.spacegroups is not None:
            raise ValueError("Set only one of 'families' or 'spacegroups', not both.")
        if self.families is not None:
            normalized = [f.capitalize() for f in self.families]
            unknown = sorted(set(normalized) - set(_ALL_FAMILIES))
            if unknown:
                raise ValueError(
                    f"Unknown crystal families {unknown}; expected one of {_ALL_FAMILIES}."
                )
            self.families = normalized
        if self.spacegroups is not None:
            invalid = [sg for sg in self.spacegroups if not 1 <= sg <= 230]
            if invalid:
                raise ValueError(
                    f"Spacegroup numbers must be in [1, 230], got {invalid}."
                )

        has_per_sg = self.structures_per_spacegroup is not None
        has_per_family = self.structures_per_family is not None
        if has_per_sg == has_per_family:
            raise ValueError(
                "Set exactly one of 'structures_per_spacegroup' or "
                "'structures_per_family'."
            )
        if has_per_sg and self.structures_per_spacegroup < 1:
            raise ValueError("structures_per_spacegroup must be >= 1.")
        if has_per_family and self.structures_per_family < 1:
            raise ValueError("structures_per_family must be >= 1.")

        if self.distribution not in ("uniform", "random"):
            raise ValueError("distribution must be 'uniform' or 'random'.")

        if self.n_species < 1:
            raise ValueError("n_species must be >= 1.")
        pool = (
            self.species_pool
            if self.species_pool is not None
            else _DEFAULT_SPECIES_POOL
        )
        if self.n_species > len(pool):
            raise ValueError(
                f"n_species={self.n_species} exceeds species_pool size ({len(pool)}); "
                "pass a larger species_pool."
            )
        if not self.candidate_num_ions:
            raise ValueError("candidate_num_ions must be non-empty.")


def _family_for_spacegroup(spacegroup: int) -> str:
    return Group(spacegroup).lattice_type.capitalize()


def _target_spacegroups(config: GenerationConfig) -> Dict[str, List[int]]:
    """``{family: [spacegroup, ...]}`` for whatever's in scope per ``config``,
    each family's list sorted ascending.
    """
    if config.spacegroups is not None:
        by_family: Dict[str, List[int]] = {}
        for sg in config.spacegroups:
            by_family.setdefault(_family_for_spacegroup(sg), []).append(sg)
        for sgs in by_family.values():
            sgs.sort()
        return by_family

    families = config.families if config.families is not None else _ALL_FAMILIES
    by_family = {family: [] for family in families}
    for sg in range(1, 231):
        family = _family_for_spacegroup(sg)
        if family in by_family:
            by_family[family].append(sg)
    return by_family


def _resolve_counts(
    config: GenerationConfig,
    by_family: Dict[str, List[int]],
    rng: np.random.Generator,
) -> Dict[int, int]:
    """``{spacegroup: how many structures to generate for it}``."""
    counts: Dict[int, int] = {}

    if config.structures_per_spacegroup is not None:
        for sgs in by_family.values():
            for sg in sgs:
                counts[sg] = config.structures_per_spacegroup
        return counts

    n = config.structures_per_family
    for sgs in by_family.values():
        if not sgs:
            continue
        if config.distribution == "uniform":
            base, extra = divmod(n, len(sgs))
            for i, sg in enumerate(sgs):
                counts[sg] = base + (1 if i < extra else 0)
        else:  # "random"
            for sg in sgs:
                counts[sg] = 0
            for _ in range(n):
                sg = sgs[int(rng.integers(len(sgs)))]
                counts[sg] += 1
    return counts


def _pick_species(
    n_species: int, pool: List[str], rng: np.random.Generator
) -> List[str]:
    indices = rng.choice(len(pool), size=n_species, replace=False)
    return [pool[i] for i in indices]


def _try_generate_one(
    spacegroup: int,
    species: List[str],
    config: GenerationConfig,
    rng: np.random.Generator,
) -> Optional[pyxtal]:
    """Retries generation across ``config.candidate_num_ions`` until one
    yields a space-group-compatible structure, or returns ``None``.
    """
    for n in config.candidate_num_ions:
        try:
            crystal = pyxtal()
            crystal.from_random(
                dim=3,
                group=spacegroup,
                species=species,
                numIons=[n] * len(species),
                factor=config.factor,
                max_count=config.max_count,
                random_state=rng,
            )
            return crystal
        except Comp_CompatibilityError:
            continue
        except Exception as exc:
            logger.debug(
                "pyxtal generation attempt failed for spacegroup %d "
                "(species=%s, numIons=%d): %s",
                spacegroup,
                species,
                n,
                exc,
            )
            continue
    return None


def generate_structures(config: GenerationConfig) -> List[Atoms]:
    """Builds a synthetic crystal-structure database with pyxtal, per ``config``.

    Returns:
        The generated ASE ``Atoms`` objects, each tagged with
        ``info["spacegroup"]`` (int), ``info["family"]`` (crystal family
        name) and ``info["material_id"]`` (``"pyxtal-<spacegroup>-<index>"``,
        mirroring ``dim_red.fetch``'s ``info["material_id"]`` convention so
        this is a drop-in alternative to it downstream).

    Warns (via the ``"dim_red.generate"`` logger) for every space group where
    one or more requested structures failed to generate, plus a final
    summary if any failures occurred at all.
    """
    rng = np.random.default_rng(config.seed)
    pool = (
        config.species_pool
        if config.species_pool is not None
        else _DEFAULT_SPECIES_POOL
    )

    by_family = _target_spacegroups(config)
    counts = _resolve_counts(config, by_family, rng)

    atoms_list: List[Atoms] = []
    failures: Dict[int, int] = {}
    material_index = 0

    for family, sgs in by_family.items():
        for sg in sgs:
            n_requested = counts.get(sg, 0)
            n_failed = 0
            for _ in range(n_requested):
                species = _pick_species(config.n_species, pool, rng)
                crystal = _try_generate_one(sg, species, config, rng)
                if crystal is None:
                    n_failed += 1
                    continue
                atoms = crystal.to_ase()
                atoms.info["spacegroup"] = sg
                atoms.info["family"] = family
                atoms.info["material_id"] = f"pyxtal-{sg}-{material_index}"
                material_index += 1
                atoms_list.append(atoms)
            if n_failed:
                failures[sg] = n_failed
                logger.warning(
                    "Spacegroup %d (%s): failed to generate %d/%d requested structure(s).",
                    sg,
                    family,
                    n_failed,
                    n_requested,
                )

    total_requested = sum(counts.values())
    if failures:
        total_failed = sum(failures.values())
        logger.warning(
            "pyxtal generation: %d/%d requested structures failed across %d "
            "spacegroup(s): %s",
            total_failed,
            total_requested,
            len(failures),
            failures,
        )
    logger.info(
        "Generated %d/%d requested structures across %d spacegroup(s).",
        len(atoms_list),
        total_requested,
        len(counts),
    )

    return atoms_list
