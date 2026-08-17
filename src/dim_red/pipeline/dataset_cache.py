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

Both data sources optionally run their structures through
``dim_red.augmentation.augment_structures`` (thermal-noise-style positional
jitter and/or vacancy removal) right after fetch/generation and before SOAP,
when ``RunConfig.augmentation`` is set -- see ``_resolve_augmentation``. The
augmentation settings are folded into the cache key (``_cache_key``/
``_pyxtal_cache_key``) so different augmentation configs don't collide.

For ``RunConfig.model_kind == "cgcnn"``, ``build_graph_dataset_for_run`` (and
its own ``get_or_build_cgcnn_dataset``/``get_or_build_pyxtal_cgcnn_dataset``)
is used *instead of* ``build_dataset_for_run`` -- it reuses the exact same
fetch/generate/augment/cache-raw-structures-to-``<hash>.extxyz`` plumbing
(all of it operates on plain ``ase.Atoms``, agnostic to downstream
featurization) but replaces SOAP with
``dim_red.cgcnn.graph.atoms_list_to_graph_arrays`` and caches a different
array schema (``_GRAPH_CACHE_ARRAY_KEYS``, keyed by graph hyperparameters
instead of SOAP ones) -- see ``dim_red.cgcnn``.
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

from dim_red.augmentation import AugmentationConfig, augment_structures
from dim_red.cgcnn.graph import atoms_list_to_graph_arrays
from dim_red.fetch import fetch_structures_by_crystal_system
from dim_red.soap import compute_soap
from dim_red.utils import apply_standardization, fit_standardization

if TYPE_CHECKING:
    from dim_red.pipeline.config import PyxtalConfig, RunConfig

logger = logging.getLogger("dim_red.pipeline")


def _augmentation_payload(
    augmentation: Optional[AugmentationConfig],
) -> Optional[Dict[str, Any]]:
    """Cache-key-safe representation of an augmentation config -- ``None``
    when augmentation is disabled, so a config that never mentions
    ``augmentation`` keeps producing the exact same cache key as before this
    feature existed.
    """
    return dataclasses.asdict(augmentation) if augmentation is not None else None


def _cache_key(
    crystal_systems: Sequence[str],
    limit_per_system: int,
    soap_kwargs: Dict[str, Any],
    augmentation: Optional[AugmentationConfig] = None,
) -> str:
    """Stable hash identifying a dataset built from (crystal systems, fetch
    limit, SOAP hyperparameters, augmentation settings).
    """
    payload = {
        "crystal_systems": sorted(cs.lower() for cs in crystal_systems),
        "limit_per_system": limit_per_system,
        "soap_kwargs": {k: soap_kwargs[k] for k in sorted(soap_kwargs)},
        "augmentation": _augmentation_payload(augmentation),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _pyxtal_cache_key(
    pyxtal_config: "PyxtalConfig",
    seed: int,
    soap_kwargs: Dict[str, Any],
    augmentation: Optional[AugmentationConfig] = None,
) -> str:
    """Stable hash identifying a dataset built from (pyxtal generation
    config, seed, SOAP hyperparameters, augmentation settings). Prefixed so
    it never collides with a ``_cache_key`` hash even if both happened to
    match numerically.
    """
    payload = {
        "pyxtal": dataclasses.asdict(pyxtal_config),
        "seed": seed,
        "soap_kwargs": {k: soap_kwargs[k] for k in sorted(soap_kwargs)},
        "augmentation": _augmentation_payload(augmentation),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return "pyxtal-" + hashlib.sha256(blob).hexdigest()[:16]


_CACHE_ARRAY_KEYS = (
    "X",
    "labels",
    "material_ids",
    "spacegroups",
    "feature_mean",
    "feature_std",
)

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
) -> Optional[
    Tuple[np.ndarray, List[str], List[str], List[int], np.ndarray, np.ndarray]
]:
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
        cached["feature_mean"],
        cached["feature_std"],
    )


def _save_dataset_cache(
    cache_path: Path,
    X_std: np.ndarray,
    labels: List[str],
    material_ids: List[str],
    spacegroups: List[int],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
) -> None:
    np.savez(
        cache_path,
        X=X_std,
        labels=np.array(labels),
        material_ids=np.array(material_ids),
        spacegroups=np.array(spacegroups, dtype=np.int64),
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    logger.info("Dataset cached at %s (shape=%s)", cache_path, X_std.shape)


def _save_structures_cache(cache_path: Path, atoms_list: list) -> None:
    structures_path = _structures_cache_path(cache_path)
    write_atoms(str(structures_path), atoms_list, format="extxyz")
    logger.info("Cached %d structures to %s", len(atoms_list), structures_path)


def _compute_soap_and_standardize(
    atoms_list: list, soap_kwargs: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns ``(X_std, feature_mean, feature_std)`` -- unlike
    ``dim_red.utils.standardize`` (fit+apply in one, discarding the fitted
    statistics), the mean/std are also returned here so they can be cached
    and, ultimately, saved into a completed run's ``embeddings.npz`` --
    letting ``dim_red.pipeline.inference.load_trained_run`` standardize new
    structures without ever recomputing SOAP on the training set again.

    All three are cast to float32 before being returned -- this is the one
    place a run's full SOAP feature matrix (potentially several GB at
    dataset sizes like 35000 structures / thousands of SOAP dimensions) gets
    produced, and both consumers already treat it as float32-precision
    anyway: nothing in this codebase enables jax's x64 mode, so `X`
    ends up truncated to float32 the moment training converts it to a jnp
    array regardless of what dtype it's stored as; `dim_red.pipeline.compare`
    (PCA/UMAP) and `dim_red.pipeline.inference` (standardizing new points)
    have no precision requirement beyond that either. Fitting mean/std in
    float64 first (inside fit_standardization/apply_standardization) and
    only downcasting the final result, rather than computing in float32
    throughout, avoids compounding rounding error across ~35000+ summed
    terms in the mean/std reduction itself.
    """
    effective_soap_kwargs = dict(soap_kwargs)
    if effective_soap_kwargs.get("species") is None:
        effective_soap_kwargs["species"] = sorted(
            {sym for a in atoms_list for sym in a.get_chemical_symbols()}
        )
    effective_soap_kwargs["average"] = "outer"

    logger.info("Computing SOAP descriptors for %d structures", len(atoms_list))
    soap_vectors = compute_soap(atoms_list, **effective_soap_kwargs)
    X = np.asarray(soap_vectors).reshape(len(atoms_list), -1)
    mean, std = fit_standardization(X)
    X_std = apply_standardization(X, mean, std)
    return X_std.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def get_or_build_dataset(
    crystal_systems: Sequence[str],
    soap_kwargs: Dict[str, Any],
    limit_per_system: int,
    cache_dir: Union[str, Path],
    api_key: Optional[str] = None,
    augmentation: Optional[AugmentationConfig] = None,
) -> Tuple[np.ndarray, List[str], List[str], List[int], Path, np.ndarray, np.ndarray]:
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
        augmentation: Optional ``dim_red.augmentation.AugmentationConfig``
            (thermal-noise-style jitter and/or vacancy removal), applied to
            each crystal system's fetched structures before they're added to
            the dataset. ``None`` (default) disables augmentation.

    Returns:
        ``(X_std, labels, material_ids, spacegroups, structures_path,
        feature_mean, feature_std)`` where ``X_std`` has shape
        ``(n_samples, n_features)``, ``spacegroups`` holds the MP spacegroup
        number (1-230) per structure (or ``-1`` when unavailable),
        ``structures_path`` is the cached extended-XYZ file holding the exact
        fetched ``Atoms`` (same order as the other arrays), and
        ``feature_mean``/``feature_std`` are the per-feature standardization
        statistics ``X_std`` was derived from (``dim_red.utils.fit_standardization``
        on the raw SOAP matrix) -- cached alongside ``X_std`` so a later
        ``dim_red.pipeline.single_run.run_single`` call can save them into its
        own ``embeddings.npz`` without recomputing SOAP, see
        ``dim_red.pipeline.inference.load_trained_run``.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(crystal_systems, limit_per_system, soap_kwargs, augmentation)
    cache_path = cache_dir / f"{key}.npz"
    structures_path = _structures_cache_path(cache_path)

    cached = _load_cached_dataset(cache_path)
    if cached is not None:
        logger.info(
            "Dataset cache hit (%s) for crystal_systems=%s", key, list(crystal_systems)
        )
        return (*cached[:4], structures_path, *cached[4:])

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
        n_fetched = len(atoms_list)
        if augmentation is not None:
            atoms_list = augment_structures(atoms_list, augmentation)
            logger.info(
                "Fetched %d structures for crystal_system=%s (%d after augmentation)",
                n_fetched,
                cs,
                len(atoms_list),
            )
        else:
            logger.info("Fetched %d structures for crystal_system=%s", n_fetched, cs)
        all_atoms.extend(atoms_list)
        labels.extend([cs.capitalize()] * len(atoms_list))

    if not all_atoms:
        raise ValueError("No structures fetched for the requested crystal systems.")

    material_ids = [a.info.get("material_id", "unknown") for a in all_atoms]
    spacegroups = [a.info.get("spacegroup", _UNKNOWN_SPACEGROUP) for a in all_atoms]

    _save_structures_cache(cache_path, all_atoms)
    X_std, feature_mean, feature_std = _compute_soap_and_standardize(
        all_atoms, soap_kwargs
    )
    _save_dataset_cache(
        cache_path, X_std, labels, material_ids, spacegroups, feature_mean, feature_std
    )

    return (
        X_std,
        labels,
        material_ids,
        spacegroups,
        structures_path,
        feature_mean,
        feature_std,
    )


def get_or_build_pyxtal_dataset(
    pyxtal_config: "PyxtalConfig",
    seed: int,
    soap_kwargs: Dict[str, Any],
    cache_dir: Union[str, Path],
    augmentation: Optional[AugmentationConfig] = None,
) -> Tuple[np.ndarray, List[str], List[str], List[int], Path, np.ndarray, np.ndarray]:
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
        augmentation: Optional ``dim_red.augmentation.AugmentationConfig``
            (thermal-noise-style jitter and/or vacancy removal), applied to
            the generated structures before SOAP. ``None`` (default) disables
            augmentation.

    Returns:
        Same shape as ``get_or_build_dataset``: ``(X_std, labels,
        material_ids, spacegroups, structures_path, feature_mean,
        feature_std)``, with ``labels`` holding each structure's crystal
        family and ``structures_path`` the cached extended-XYZ file holding
        the exact generated ``Atoms``.
    """
    # Lazy: dim_red.generate requires pyxtal, not a hard dim_red dependency.
    from dim_red.generate import GenerationConfig, generate_structures

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _pyxtal_cache_key(pyxtal_config, seed, soap_kwargs, augmentation)
    cache_path = cache_dir / f"{key}.npz"
    structures_path = _structures_cache_path(cache_path)

    cached = _load_cached_dataset(cache_path)
    if cached is not None:
        logger.info("Dataset cache hit (%s) for pyxtal generation", key)
        return (*cached[:4], structures_path, *cached[4:])

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

    n_generated = len(all_atoms)
    if augmentation is not None:
        all_atoms = augment_structures(all_atoms, augmentation)
        logger.info(
            "Generated %d structures with pyxtal (%d after augmentation)",
            n_generated,
            len(all_atoms),
        )

    labels = [a.info["family"] for a in all_atoms]
    material_ids = [a.info["material_id"] for a in all_atoms]
    spacegroups = [a.info["spacegroup"] for a in all_atoms]

    _save_structures_cache(cache_path, all_atoms)
    X_std, feature_mean, feature_std = _compute_soap_and_standardize(
        all_atoms, soap_kwargs
    )
    _save_dataset_cache(
        cache_path, X_std, labels, material_ids, spacegroups, feature_mean, feature_std
    )

    return (
        X_std,
        labels,
        material_ids,
        spacegroups,
        structures_path,
        feature_mean,
        feature_std,
    )


def _resolve_augmentation(config: "RunConfig") -> Optional[AugmentationConfig]:
    """Converts ``RunConfig.augmentation`` (``dim_red.pipeline.config``'s
    own, ``dim_red.augmentation``-independent dataclass -- see that class's
    docstring) into the real ``dim_red.augmentation.AugmentationConfig`` that
    ``get_or_build_dataset``/``get_or_build_pyxtal_dataset`` actually use.
    ``None`` when augmentation is disabled. Resolves ``seed`` the same way
    ``build_dataset_for_run`` already does for ``config.pyxtal.seed``: falls
    back to ``config.seed`` when ``config.augmentation.seed`` is ``None``.
    """
    if config.augmentation is None:
        return None
    seed = (
        config.augmentation.seed
        if config.augmentation.seed is not None
        else config.seed
    )
    return AugmentationConfig(
        n_augmented=config.augmentation.n_augmented,
        keep_original=config.augmentation.keep_original,
        jitter_probability=config.augmentation.jitter_probability,
        jitter_std=config.augmentation.jitter_std,
        vacancy_probability=config.augmentation.vacancy_probability,
        vacancy_atom_probability=config.augmentation.vacancy_atom_probability,
        max_vacancies=config.augmentation.max_vacancies,
        supercell_radius=config.augmentation.supercell_radius,
        seed=seed,
    )


def build_dataset_for_run(
    config: "RunConfig", cache_dir: Union[str, Path]
) -> Tuple[np.ndarray, List[str], List[str], List[int], Path, np.ndarray, np.ndarray]:
    """Dispatches to ``get_or_build_dataset`` or ``get_or_build_pyxtal_dataset``
    per ``config.data_source`` -- the single entry point
    ``dim_red.pipeline.single_run.run_single`` uses, so callers don't need to
    know which data source a run uses. Same 7-element return shape as both:
    ``(X_std, labels, material_ids, spacegroups, structures_path,
    feature_mean, feature_std)``.
    """
    augmentation = _resolve_augmentation(config)
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
            augmentation=augmentation,
        )
    return get_or_build_dataset(
        crystal_systems=config.fetch.crystal_systems,
        soap_kwargs=config.soap.as_kwargs(),
        limit_per_system=config.fetch.limit_per_system,
        cache_dir=cache_dir,
        api_key=config.fetch.api_key,
        augmentation=augmentation,
    )


# --- CGCNN graph dataset caching (RunConfig.model_kind == "cgcnn" only) ----
#
# Parallel to the SOAP-based functions above: the fetch/generate/augment/
# cache-raw-structures-to-<hash>.extxyz plumbing is reused unchanged (it
# never depends on downstream featurization); only feature computation and
# the cache array schema differ (graph arrays instead of a flat SOAP
# matrix -- no standardization step, since Gaussian-expanded distances are
# already bounded to [0, 1] by construction).

_GRAPH_CACHE_ARRAY_KEYS = (
    "local_species_idx",
    "nbr_idx",
    "nbr_fea",
    "nbr_mask",
    "atom_mask",
    "labels",
    "material_ids",
    "spacegroups",
)


def _graph_cache_key(
    crystal_systems: Sequence[str],
    limit_per_system: int,
    graph_kwargs: Dict[str, Any],
    augmentation: Optional[AugmentationConfig] = None,
) -> str:
    """Stable hash identifying a graph dataset built from (crystal systems,
    fetch limit, graph-construction hyperparameters, augmentation settings).
    Prefixed ``"cgcnn-"`` so it can never collide with a ``_cache_key``
    (SOAP) hash for the same underlying dataset-defining scope even if the
    hashes matched numerically.
    """
    payload = {
        "crystal_systems": sorted(cs.lower() for cs in crystal_systems),
        "limit_per_system": limit_per_system,
        "graph_kwargs": {k: graph_kwargs[k] for k in sorted(graph_kwargs)},
        "augmentation": _augmentation_payload(augmentation),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return "cgcnn-" + hashlib.sha256(blob).hexdigest()[:16]


def _pyxtal_graph_cache_key(
    pyxtal_config: "PyxtalConfig",
    seed: int,
    graph_kwargs: Dict[str, Any],
    augmentation: Optional[AugmentationConfig] = None,
) -> str:
    """Stable hash identifying a graph dataset built from (pyxtal generation
    config, seed, graph-construction hyperparameters, augmentation
    settings). Prefixed ``"cgcnn-pyxtal-"``, mirrors ``_pyxtal_cache_key``.
    """
    payload = {
        "pyxtal": dataclasses.asdict(pyxtal_config),
        "seed": seed,
        "graph_kwargs": {k: graph_kwargs[k] for k in sorted(graph_kwargs)},
        "augmentation": _augmentation_payload(augmentation),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return "cgcnn-pyxtal-" + hashlib.sha256(blob).hexdigest()[:16]


def _load_cached_graph_dataset(
    cache_path: Path,
) -> Optional[
    Tuple[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        List[str],
        List[str],
        List[int],
    ]
]:
    """Mirrors ``_load_cached_dataset``: ``None`` on cache miss/stale schema,
    otherwise ``((local_species_idx, nbr_idx, nbr_fea, nbr_mask, atom_mask),
    labels, material_ids, spacegroups)``.
    """
    if not cache_path.exists():
        return None
    if not _structures_cache_path(cache_path).exists():
        logger.info(
            "Graph dataset cache at %s has no cached structures (%s); rebuilding",
            cache_path,
            _structures_cache_path(cache_path),
        )
        return None
    cached = np.load(cache_path)
    if not all(k in cached.files for k in _GRAPH_CACHE_ARRAY_KEYS):
        logger.info(
            "Graph dataset cache at %s is missing expected fields (stale schema); rebuilding",
            cache_path,
        )
        return None
    graph_arrays = (
        cached["local_species_idx"],
        cached["nbr_idx"],
        cached["nbr_fea"],
        cached["nbr_mask"],
        cached["atom_mask"],
    )
    return (
        graph_arrays,
        cached["labels"].tolist(),
        cached["material_ids"].tolist(),
        cached["spacegroups"].tolist(),
    )


def _save_graph_dataset_cache(
    cache_path: Path,
    local_species_idx: np.ndarray,
    nbr_idx: np.ndarray,
    nbr_fea: np.ndarray,
    nbr_mask: np.ndarray,
    atom_mask: np.ndarray,
    labels: List[str],
    material_ids: List[str],
    spacegroups: List[int],
) -> None:
    """Mirrors ``_save_dataset_cache`` for the graph array schema."""
    np.savez(
        cache_path,
        local_species_idx=local_species_idx,
        nbr_idx=nbr_idx,
        nbr_fea=nbr_fea,
        nbr_mask=nbr_mask,
        atom_mask=atom_mask,
        labels=np.array(labels),
        material_ids=np.array(material_ids),
        spacegroups=np.array(spacegroups, dtype=np.int64),
    )
    logger.info(
        "Graph dataset cached at %s (local_species_idx.shape=%s)",
        cache_path,
        local_species_idx.shape,
    )


def _compute_graphs(
    atoms_list: list, graph_kwargs: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns ``(local_species_idx, nbr_idx, nbr_fea, nbr_mask, atom_mask)``
    for the whole dataset -- the CGCNN counterpart to
    ``_compute_soap_and_standardize``, used *instead of* it when
    ``RunConfig.model_kind == "cgcnn"``. Unlike SOAP, there is no
    standardization step (Gaussian-expanded distances are already bounded
    to ``[0, 1]`` by construction) and no species list needs resolving (the
    model uses a fixed-size embedding keyed by a per-structure LOCAL species
    slot, not a real chemical species vocabulary -- see
    ``dim_red.cgcnn.graph``). ``max_atoms`` is auto-resolved across this
    ``atoms_list`` (dataset-wide).
    """
    logger.info("Computing CGCNN graphs for %d structures", len(atoms_list))
    return atoms_list_to_graph_arrays(atoms_list, **graph_kwargs)


def get_or_build_cgcnn_dataset(
    crystal_systems: Sequence[str],
    graph_kwargs: Dict[str, Any],
    limit_per_system: int,
    cache_dir: Union[str, Path],
    api_key: Optional[str] = None,
    augmentation: Optional[AugmentationConfig] = None,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    List[str],
    List[str],
    List[int],
    Path,
]:
    """Fetch structures and build CGCNN graph arrays -- reusing a cached copy
    on disk when available. The graph-based counterpart to
    ``get_or_build_dataset``; the fetch/augment/cache-structures plumbing is
    reused unchanged (see module docstring), only the featurization step and
    cache-array schema differ.

    Args:
        crystal_systems: Crystal systems to include (as accepted by
            ``fetch_structures_by_crystal_system``).
        graph_kwargs: Keyword arguments for ``atoms_list_to_graph_arrays``
            (``dim_red.pipeline.config.GraphConfig.graph_kwargs()``).
        limit_per_system: Max structures fetched per crystal system.
        cache_dir: Directory where cached datasets (``<hash>.npz``/
            ``<hash>.extxyz``) live.
        api_key: Materials Project API key (falls back to ``MP_API_KEY``).
        augmentation: Optional ``dim_red.augmentation.AugmentationConfig``,
            applied before graph construction. ``None`` (default) disables it.

    Returns:
        ``(graph_arrays, labels, material_ids, spacegroups, structures_path)``
        -- ``graph_arrays`` is the 5-tuple ``(local_species_idx, nbr_idx,
        nbr_fea, nbr_mask, atom_mask)``. Note: 5 elements, not 7 like
        ``get_or_build_dataset`` -- no ``feature_mean``/``feature_std`` (not
        applicable to graph features).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _graph_cache_key(
        crystal_systems, limit_per_system, graph_kwargs, augmentation
    )
    cache_path = cache_dir / f"{key}.npz"
    structures_path = _structures_cache_path(cache_path)

    cached = _load_cached_graph_dataset(cache_path)
    if cached is not None:
        logger.info(
            "CGCNN dataset cache hit (%s) for crystal_systems=%s",
            key,
            list(crystal_systems),
        )
        graph_arrays, labels, material_ids, spacegroups = cached
        return graph_arrays, labels, material_ids, spacegroups, structures_path

    logger.info(
        "CGCNN dataset cache miss (%s); fetching structures for crystal_systems=%s",
        key,
        list(crystal_systems),
    )

    all_atoms = []
    labels: List[str] = []
    for cs in crystal_systems:
        atoms_list = fetch_structures_by_crystal_system(
            crystal_system=cs, api_key=api_key, limit=limit_per_system
        )
        n_fetched = len(atoms_list)
        if augmentation is not None:
            atoms_list = augment_structures(atoms_list, augmentation)
            logger.info(
                "Fetched %d structures for crystal_system=%s (%d after augmentation)",
                n_fetched,
                cs,
                len(atoms_list),
            )
        else:
            logger.info("Fetched %d structures for crystal_system=%s", n_fetched, cs)
        all_atoms.extend(atoms_list)
        labels.extend([cs.capitalize()] * len(atoms_list))

    if not all_atoms:
        raise ValueError("No structures fetched for the requested crystal systems.")

    material_ids = [a.info.get("material_id", "unknown") for a in all_atoms]
    spacegroups = [a.info.get("spacegroup", _UNKNOWN_SPACEGROUP) for a in all_atoms]

    _save_structures_cache(cache_path, all_atoms)
    local_species_idx, nbr_idx, nbr_fea, nbr_mask, atom_mask = _compute_graphs(
        all_atoms, graph_kwargs
    )
    _save_graph_dataset_cache(
        cache_path,
        local_species_idx,
        nbr_idx,
        nbr_fea,
        nbr_mask,
        atom_mask,
        labels,
        material_ids,
        spacegroups,
    )

    return (
        (local_species_idx, nbr_idx, nbr_fea, nbr_mask, atom_mask),
        labels,
        material_ids,
        spacegroups,
        structures_path,
    )


def get_or_build_pyxtal_cgcnn_dataset(
    pyxtal_config: "PyxtalConfig",
    seed: int,
    graph_kwargs: Dict[str, Any],
    cache_dir: Union[str, Path],
    augmentation: Optional[AugmentationConfig] = None,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    List[str],
    List[str],
    List[int],
    Path,
]:
    """Generate a synthetic structure database with ``dim_red.generate`` and
    build CGCNN graph arrays -- reusing a cached copy on disk when
    available. The graph-based, ``pyxtal`` counterpart to
    ``get_or_build_pyxtal_dataset``.

    Args:
        pyxtal_config: Generation settings (``dim_red.pipeline.config.PyxtalConfig``).
        seed: The *effective* seed to generate with -- see
            ``get_or_build_pyxtal_dataset``'s ``seed`` docs; resolving
            ``pyxtal_config.seed or RunConfig.seed`` is the caller's job
            (``build_graph_dataset_for_run`` does this).
        graph_kwargs: Keyword arguments for ``atoms_list_to_graph_arrays``.
        cache_dir: Directory where cached datasets (``<hash>.npz``/
            ``<hash>.extxyz``) live.
        augmentation: Optional ``dim_red.augmentation.AugmentationConfig``,
            applied before graph construction. ``None`` (default) disables it.

    Returns:
        Same shape as ``get_or_build_cgcnn_dataset``: ``(graph_arrays,
        labels, material_ids, spacegroups, structures_path)``.
    """
    # Lazy: dim_red.generate requires pyxtal, not a hard dim_red dependency.
    from dim_red.generate import GenerationConfig, generate_structures

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _pyxtal_graph_cache_key(pyxtal_config, seed, graph_kwargs, augmentation)
    cache_path = cache_dir / f"{key}.npz"
    structures_path = _structures_cache_path(cache_path)

    cached = _load_cached_graph_dataset(cache_path)
    if cached is not None:
        logger.info("CGCNN dataset cache hit (%s) for pyxtal generation", key)
        graph_arrays, labels, material_ids, spacegroups = cached
        return graph_arrays, labels, material_ids, spacegroups, structures_path

    logger.info("CGCNN dataset cache miss (%s); generating structures with pyxtal", key)

    generation_kwargs = dataclasses.asdict(pyxtal_config)
    generation_kwargs.pop("seed", None)  # the resolved "seed" arg wins
    if generation_kwargs.get("candidate_num_ions") is None:
        generation_kwargs.pop("candidate_num_ions")
    all_atoms = generate_structures(GenerationConfig(seed=seed, **generation_kwargs))

    if not all_atoms:
        raise ValueError(
            "pyxtal generated no structures for the requested configuration."
        )

    n_generated = len(all_atoms)
    if augmentation is not None:
        all_atoms = augment_structures(all_atoms, augmentation)
        logger.info(
            "Generated %d structures with pyxtal (%d after augmentation)",
            n_generated,
            len(all_atoms),
        )

    labels = [a.info["family"] for a in all_atoms]
    material_ids = [a.info["material_id"] for a in all_atoms]
    spacegroups = [a.info["spacegroup"] for a in all_atoms]

    _save_structures_cache(cache_path, all_atoms)
    local_species_idx, nbr_idx, nbr_fea, nbr_mask, atom_mask = _compute_graphs(
        all_atoms, graph_kwargs
    )
    _save_graph_dataset_cache(
        cache_path,
        local_species_idx,
        nbr_idx,
        nbr_fea,
        nbr_mask,
        atom_mask,
        labels,
        material_ids,
        spacegroups,
    )

    return (
        (local_species_idx, nbr_idx, nbr_fea, nbr_mask, atom_mask),
        labels,
        material_ids,
        spacegroups,
        structures_path,
    )


def build_graph_dataset_for_run(
    config: "RunConfig", cache_dir: Union[str, Path]
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    List[str],
    List[str],
    List[int],
    Path,
]:
    """Dispatches to ``get_or_build_cgcnn_dataset`` or
    ``get_or_build_pyxtal_cgcnn_dataset`` per ``config.data_source`` --
    the graph-based counterpart to ``build_dataset_for_run``, called by
    ``dim_red.pipeline.single_run.run_single`` only when
    ``config.model_kind == "cgcnn"``. Additive: ``build_dataset_for_run``'s
    own contract is untouched. Returns the 5-element graph-shaped tuple
    ``(graph_arrays, labels, material_ids, spacegroups, structures_path)``
    instead of the 7-element SOAP-shaped one.
    """
    augmentation = _resolve_augmentation(config)
    graph_kwargs = config.graph.graph_kwargs()
    if config.data_source == "pyxtal":
        seed = config.pyxtal.seed if config.pyxtal.seed is not None else config.seed
        return get_or_build_pyxtal_cgcnn_dataset(
            pyxtal_config=config.pyxtal,
            seed=seed,
            graph_kwargs=graph_kwargs,
            cache_dir=cache_dir,
            augmentation=augmentation,
        )
    return get_or_build_cgcnn_dataset(
        crystal_systems=config.fetch.crystal_systems,
        graph_kwargs=graph_kwargs,
        limit_per_system=config.fetch.limit_per_system,
        cache_dir=cache_dir,
        api_key=config.fetch.api_key,
        augmentation=augmentation,
    )
