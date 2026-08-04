"""
Caching layer for the dataset -> SOAP -> standardize construction step, so
that a hidden-layer sweep (axis A) does not repeat the (Materials Project
fetch or pyxtal generation) + SOAP computation for every combination that
shares the same dataset-defining subset (axis B).

Two data sources are supported, selected by ``RunConfig.data_source``:
``get_or_build_dataset`` (``"fetch"``, Materials Project) and
``get_or_build_pyxtal_dataset`` (``"pyxtal"``, synthetic generation via
``dim_red.generate`` -- imported lazily, only when actually used, since
``pyxtal`` isn't a hard dim_red dependency). ``build_dataset_for_run``
dispatches between the two from a ``RunConfig``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from ase.io import write as write_atoms

from dim_red.fetch import fetch_structures_by_crystal_system
from dim_red.soap import compute_soap
from dim_red.utils import standardize

if TYPE_CHECKING:
    from dim_red.pipeline.config import PyxtalConfig, RunConfig

logger = logging.getLogger("dim_red.pipeline")


def _cache_key(
    crystal_systems: Sequence[str], limit_per_system: int, soap_kwargs: Dict[str, Any]
) -> str:
    """Stable hash identifying a dataset built from (crystal systems, fetch
    limit, SOAP hyperparameters).
    """
    payload = {
        "crystal_systems": sorted(cs.lower() for cs in crystal_systems),
        "limit_per_system": limit_per_system,
        "soap_kwargs": {k: soap_kwargs[k] for k in sorted(soap_kwargs)},
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _pyxtal_cache_key(
    pyxtal_config: "PyxtalConfig", seed: int, soap_kwargs: Dict[str, Any]
) -> str:
    """Stable hash identifying a dataset built from (pyxtal generation
    config, seed, SOAP hyperparameters). Prefixed so it never collides with
    a ``_cache_key`` hash even if both happened to match numerically.
    """
    payload = {
        "pyxtal": dataclasses.asdict(pyxtal_config),
        "seed": seed,
        "soap_kwargs": {k: soap_kwargs[k] for k in sorted(soap_kwargs)},
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return "pyxtal-" + hashlib.sha256(blob).hexdigest()[:16]


_CACHE_ARRAY_KEYS = ("X", "labels", "material_ids", "spacegroups")

# Sentinel spacegroup number for structures MP didn't return symmetry data for.
_UNKNOWN_SPACEGROUP = -1


def _structures_cache_path(cache_path: Path) -> Path:
    """The raw-structures (extended XYZ) cache file sitting next to a
    dataset's ``<hash>.npz``, so ``dim_red.pipeline.single_run.run_single``
    can copy the exact structures used into each run directory without
    holding every ``Atoms`` object across a cache hit.
    """
    return cache_path.with_suffix(".extxyz")


def _load_cached_dataset(
    cache_path: Path,
) -> Optional[Tuple[np.ndarray, List[str], List[str], List[int]]]:
    if not cache_path.exists():
        return None
    if not _structures_cache_path(cache_path).exists():
        logger.info(
            "Dataset cache at %s has no cached structures (%s); rebuilding",
            cache_path,
            _structures_cache_path(cache_path),
        )
        return None
    cached = np.load(cache_path)
    if not all(k in cached.files for k in _CACHE_ARRAY_KEYS):
        logger.info(
            "Dataset cache at %s is missing expected fields (stale schema); rebuilding",
            cache_path,
        )
        return None
    return (
        cached["X"],
        cached["labels"].tolist(),
        cached["material_ids"].tolist(),
        cached["spacegroups"].tolist(),
    )


def _save_dataset_cache(
    cache_path: Path,
    X_std: np.ndarray,
    labels: List[str],
    material_ids: List[str],
    spacegroups: List[int],
) -> None:
    np.savez(
        cache_path,
        X=X_std,
        labels=np.array(labels),
        material_ids=np.array(material_ids),
        spacegroups=np.array(spacegroups, dtype=np.int64),
    )
    logger.info("Dataset cached at %s (shape=%s)", cache_path, X_std.shape)


def _save_structures_cache(cache_path: Path, atoms_list: list) -> None:
    structures_path = _structures_cache_path(cache_path)
    write_atoms(str(structures_path), atoms_list, format="extxyz")
    logger.info("Cached %d structures to %s", len(atoms_list), structures_path)


def _compute_soap_and_standardize(
    atoms_list: list, soap_kwargs: Dict[str, Any]
) -> np.ndarray:
    effective_soap_kwargs = dict(soap_kwargs)
    if effective_soap_kwargs.get("species") is None:
        effective_soap_kwargs["species"] = sorted(
            {sym for a in atoms_list for sym in a.get_chemical_symbols()}
        )
    effective_soap_kwargs["average"] = "outer"

    logger.info("Computing SOAP descriptors for %d structures", len(atoms_list))
    soap_vectors = compute_soap(atoms_list, **effective_soap_kwargs)
    X = np.asarray(soap_vectors).reshape(len(atoms_list), -1)
    return standardize(X)


def get_or_build_dataset(
    crystal_systems: Sequence[str],
    soap_kwargs: Dict[str, Any],
    limit_per_system: int,
    cache_dir: Union[str, Path],
    api_key: Optional[str] = None,
) -> Tuple[np.ndarray, List[str], List[str], List[int], Path]:
    """Fetch structures, compute a global SOAP descriptor per structure, and
    standardize the result -- reusing a cached copy on disk when available.

    Args:
        crystal_systems: Crystal systems to include (as accepted by
            ``fetch_structures_by_crystal_system``).
        soap_kwargs: Keyword arguments for ``compute_soap`` minus ``average``
            (the pipeline always requests ``average="outer"``).
        limit_per_system: Max structures fetched per crystal system.
        cache_dir: Directory where cached datasets (``<hash>.npz``/
            ``<hash>.extxyz``) live.
        api_key: Materials Project API key (falls back to ``MP_API_KEY``).

    Returns:
        ``(X_std, labels, material_ids, spacegroups, structures_path)``
        where ``X_std`` has shape ``(n_samples, n_features)``, ``spacegroups``
        holds the MP spacegroup number (1-230) per structure (or ``-1`` when
        unavailable), and ``structures_path`` is the cached extended-XYZ file
        holding the exact fetched ``Atoms`` (same order as the other arrays).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(crystal_systems, limit_per_system, soap_kwargs)
    cache_path = cache_dir / f"{key}.npz"
    structures_path = _structures_cache_path(cache_path)

    cached = _load_cached_dataset(cache_path)
    if cached is not None:
        logger.info(
            "Dataset cache hit (%s) for crystal_systems=%s", key, list(crystal_systems)
        )
        return (*cached, structures_path)

    logger.info(
        "Dataset cache miss (%s); fetching structures for crystal_systems=%s",
        key,
        list(crystal_systems),
    )

    all_atoms = []
    labels: List[str] = []
    for cs in crystal_systems:
        atoms_list = fetch_structures_by_crystal_system(
            crystal_system=cs, api_key=api_key, limit=limit_per_system
        )
        logger.info("Fetched %d structures for crystal_system=%s", len(atoms_list), cs)
        all_atoms.extend(atoms_list)
        labels.extend([cs.capitalize()] * len(atoms_list))

    if not all_atoms:
        raise ValueError("No structures fetched for the requested crystal systems.")

    material_ids = [a.info.get("material_id", "unknown") for a in all_atoms]
    spacegroups = [a.info.get("spacegroup", _UNKNOWN_SPACEGROUP) for a in all_atoms]

    _save_structures_cache(cache_path, all_atoms)
    X_std = _compute_soap_and_standardize(all_atoms, soap_kwargs)
    _save_dataset_cache(cache_path, X_std, labels, material_ids, spacegroups)

    return X_std, labels, material_ids, spacegroups, structures_path


def get_or_build_pyxtal_dataset(
    pyxtal_config: "PyxtalConfig",
    seed: int,
    soap_kwargs: Dict[str, Any],
    cache_dir: Union[str, Path],
) -> Tuple[np.ndarray, List[str], List[str], List[int], Path]:
    """Generate a synthetic structure database with ``dim_red.generate``,
    compute a global SOAP descriptor per structure, and standardize the
    result -- reusing a cached copy on disk when available. The ``pyxtal``
    (and ``jax``-free) counterpart to ``get_or_build_dataset``.

    Args:
        pyxtal_config: Generation settings (``dim_red.pipeline.config.PyxtalConfig``).
        seed: The *effective* seed to generate with -- reused as
            ``dim_red.generate.GenerationConfig.seed``, taking priority over
            (and ignoring) ``pyxtal_config.seed``; resolving
            ``pyxtal_config.seed or RunConfig.seed`` is the caller's job
            (``build_dataset_for_run`` does this).
        soap_kwargs: Keyword arguments for ``compute_soap`` minus ``average``.
        cache_dir: Directory where cached datasets (``<hash>.npz``/
            ``<hash>.extxyz``) live.

    Returns:
        Same shape as ``get_or_build_dataset``: ``(X_std, labels,
        material_ids, spacegroups, structures_path)``, with ``labels``
        holding each structure's crystal family and ``structures_path`` the
        cached extended-XYZ file holding the exact generated ``Atoms``.
    """
    # Lazy: dim_red.generate requires pyxtal, not a hard dim_red dependency.
    from dim_red.generate import GenerationConfig, generate_structures

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _pyxtal_cache_key(pyxtal_config, seed, soap_kwargs)
    cache_path = cache_dir / f"{key}.npz"
    structures_path = _structures_cache_path(cache_path)

    cached = _load_cached_dataset(cache_path)
    if cached is not None:
        logger.info("Dataset cache hit (%s) for pyxtal generation", key)
        return (*cached, structures_path)

    logger.info("Dataset cache miss (%s); generating structures with pyxtal", key)

    generation_kwargs = dataclasses.asdict(pyxtal_config)
    generation_kwargs.pop("seed", None)  # the resolved "seed" arg wins
    if generation_kwargs.get("candidate_num_ions") is None:
        generation_kwargs.pop("candidate_num_ions")
    all_atoms = generate_structures(GenerationConfig(seed=seed, **generation_kwargs))

    if not all_atoms:
        raise ValueError(
            "pyxtal generated no structures for the requested configuration."
        )

    labels = [a.info["family"] for a in all_atoms]
    material_ids = [a.info["material_id"] for a in all_atoms]
    spacegroups = [a.info["spacegroup"] for a in all_atoms]

    _save_structures_cache(cache_path, all_atoms)
    X_std = _compute_soap_and_standardize(all_atoms, soap_kwargs)
    _save_dataset_cache(cache_path, X_std, labels, material_ids, spacegroups)

    return X_std, labels, material_ids, spacegroups, structures_path


def build_dataset_for_run(
    config: "RunConfig", cache_dir: Union[str, Path]
) -> Tuple[np.ndarray, List[str], List[str], List[int], Path]:
    """Dispatches to ``get_or_build_dataset`` or ``get_or_build_pyxtal_dataset``
    per ``config.data_source`` -- the single entry point
    ``dim_red.pipeline.single_run.run_single`` uses, so callers don't need to
    know which data source a run uses. Same 5-element return shape as both:
    ``(X_std, labels, material_ids, spacegroups, structures_path)``.
    """
    if config.data_source == "pyxtal":
        # config.pyxtal.seed (if set) pins the generated dataset independently
        # of config.seed, so sweeping config.seed (e.g. repeated-seed training
        # runs) or other hyperparameters doesn't change the dataset underneath.
        seed = config.pyxtal.seed if config.pyxtal.seed is not None else config.seed
        return get_or_build_pyxtal_dataset(
            pyxtal_config=config.pyxtal,
            seed=seed,
            soap_kwargs=config.soap.as_kwargs(),
            cache_dir=cache_dir,
        )
    return get_or_build_dataset(
        crystal_systems=config.fetch.crystal_systems,
        soap_kwargs=config.soap.as_kwargs(),
        limit_per_system=config.fetch.limit_per_system,
        cache_dir=cache_dir,
        api_key=config.fetch.api_key,
    )
