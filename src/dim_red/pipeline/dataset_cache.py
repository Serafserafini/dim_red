"""
Caching layer for the dataset -> features construction step, so that a sweep
does not repeat the (Materials Project fetch or pyxtal generation) +
featurization for every combination that shares the same dataset-defining
settings.

Two structure sources are supported: Materials Project fetch
(``get_or_build_dataset``/``get_or_build_cgcnn_dataset``/
``get_or_build_mace_dataset``) and synthetic pyxtal generation
(``get_or_build_pyxtal_dataset``/``get_or_build_pyxtal_cgcnn_dataset``/
``get_or_build_pyxtal_mace_dataset``, via ``dim_red.generate`` -- imported
lazily, only when actually used, since ``pyxtal`` isn't a hard dim_red
dependency). FullStack (``dim_red.pipeline.stack_data``) uses the pyxtal
SOAP/MACE builders; cgcnn's ``run_single`` uses ``build_graph_dataset_for_run``,
which dispatches between the two graph builders from a ``RunConfig``.

Every builder optionally runs its structures through
``dim_red.augmentation.augment_structures`` (thermal-noise-style positional
jitter and/or vacancy removal) right after fetch/generation and before
featurization -- see ``resolve_augmentation``. The augmentation settings are
folded into the cache key (``_cache_key``/``_pyxtal_cache_key``/...) so
different augmentation configs don't collide.

The cgcnn graph builders reuse the exact same
fetch/generate/augment/cache-raw-structures-to-``<hash>.extxyz`` plumbing
(all of it operates on plain ``ase.Atoms``, agnostic to downstream
featurization) but replace SOAP with
``dim_red.cgcnn.graph.atoms_list_to_graph_arrays`` and cache a different
array schema (``_GRAPH_CACHE_ARRAY_KEYS``, keyed by graph hyperparameters
instead of SOAP ones) -- see ``dim_red.cgcnn``.

All six ``get_or_build_*`` functions are thin wrappers around the single
``_get_or_build`` sequence, parameterized by a ``_Featurizer`` (SOAP, CGCNN
graph or MACE) and a ``_StructureSource`` (Materials Project fetch or pyxtal
generation).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from ase.io import write as write_atoms

from dim_red.augmentation import AugmentationConfig, augment_structures
from dim_red.cgcnn.graph import atoms_list_to_graph_arrays
from dim_red.fetch import fetch_structures_by_crystal_system
from dim_red.soap import compute_soap
from dim_red.utils import apply_standardization, fit_standardization

if TYPE_CHECKING:
    from dim_red.pipeline.config import AugmentationConfig as PipelineAugmentationConfig
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
    if augmentation is None:
        return None
    payload = dataclasses.asdict(augmentation)
    # ``jitter_std_relative`` no longer exists, but every cache built while it
    # did hashed it (always False in practice) -- keep it in the payload so
    # those existing, potentially multi-GB caches stay valid.
    payload["jitter_std_relative"] = False
    return payload


def _soap_kwargs_payload(soap_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Cache-key-safe SOAP kwargs: sorted, plus the removed
    ``normalize_distances`` option pinned to its old default (``False``) so
    caches built while that option existed keep the same hash.
    """
    payload = {k: soap_kwargs[k] for k in sorted(soap_kwargs)}
    payload["normalize_distances"] = False
    return payload


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
        "soap_kwargs": _soap_kwargs_payload(soap_kwargs),
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
        "soap_kwargs": _soap_kwargs_payload(soap_kwargs),
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
    alongside the features and reused to standardize new structures the
    same way, without ever recomputing SOAP on the training set again.

    All three are cast to float32 before being returned -- this is the one
    place a run's full SOAP feature matrix (potentially several GB at
    dataset sizes like 35000 structures / thousands of SOAP dimensions) gets
    produced, and both consumers already treat it as float32-precision
    anyway: nothing in this codebase enables jax's x64 mode, so `X`
    ends up truncated to float32 the moment training converts it to a jnp
    array regardless of what dtype it's stored as; `dim_red.pipeline.compare`
    (PCA/UMAP) and standardizing new points have no precision requirement
    beyond that either. Fitting mean/std in
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
            applied to each crystal system's fetched structures. ``None``
            (default) disables augmentation.

    Returns:
        ``(X_std, labels, material_ids, spacegroups, structures_path,
        feature_mean, feature_std)``: ``X_std`` has shape
        ``(n_samples, n_features)``, ``spacegroups`` holds the MP spacegroup
        number (``-1`` when unavailable), ``structures_path`` is the cached
        extended-XYZ file with the exact ``Atoms`` (same order), and
        ``feature_mean``/``feature_std`` are the standardization statistics
        ``X_std`` was derived from, cached so they never need to be
        recomputed.
    """
    key = _cache_key(crystal_systems, limit_per_system, soap_kwargs, augmentation)
    return _get_or_build(
        _soap_featurizer(),
        _fetch_source(crystal_systems, limit_per_system, api_key),
        key,
        soap_kwargs,
        cache_dir,
        augmentation,
    )


def get_or_build_pyxtal_dataset(
    pyxtal_config: "PyxtalConfig",
    seed: int,
    soap_kwargs: Dict[str, Any],
    cache_dir: Union[str, Path],
    augmentation: Optional[AugmentationConfig] = None,
) -> Tuple[np.ndarray, List[str], List[str], List[int], Path, np.ndarray, np.ndarray]:
    """The ``pyxtal`` counterpart to ``get_or_build_dataset``: generate a
    synthetic structure database with ``dim_red.generate``, compute and
    standardize SOAP descriptors, reusing a cached copy when available.

    Args:
        pyxtal_config: Generation settings (``dim_red.pipeline.config.PyxtalConfig``).
        seed: The *effective* seed to generate with, taking priority over
            ``pyxtal_config.seed`` (resolving ``pyxtal_config.seed or
            RunConfig.seed`` is the caller's job, e.g.
            ``dim_red.pipeline.stack_data``).
        soap_kwargs: Keyword arguments for ``compute_soap`` minus ``average``.
        cache_dir: Directory where cached datasets live.
        augmentation: Optional ``dim_red.augmentation.AugmentationConfig``
            applied to the generated structures before SOAP.

    Returns:
        Same 7-element shape as ``get_or_build_dataset``, with ``labels``
        holding each structure's crystal family.
    """
    key = _pyxtal_cache_key(pyxtal_config, seed, soap_kwargs, augmentation)
    return _get_or_build(
        _soap_featurizer(),
        _pyxtal_source(pyxtal_config, seed),
        key,
        soap_kwargs,
        cache_dir,
        augmentation,
    )


# --- Shared get-or-build sequence ------------------------------------------
#
# Every ``get_or_build_*`` function below follows the same steps (compute the
# cache key, try to load, otherwise obtain structures, augment, cache the
# structures, featurize, cache the arrays, return); only the featurizer and
# the structure source differ. Both are small objects looked up through module
# globals at call time (the lambdas), so tests patching e.g.
# ``dataset_cache.compute_soap`` or ``_compute_mace_and_standardize`` keep
# working.


@dataclasses.dataclass(frozen=True)
class _Featurizer:
    """How one feature kind (SOAP / CGCNN graph / MACE) is computed, cached
    and returned.

    Attributes:
        log_prefix: Start of the cache hit/miss log lines (e.g. ``"Dataset"``).
        load: ``cache_path -> (features, labels, material_ids, spacegroups)``
            or ``None`` on a cache miss/stale schema. ``features`` is opaque
            to the shared sequence.
        compute: ``(atoms_list, kwargs) -> features``.
        save: ``(cache_path, features, labels, material_ids, spacegroups)``.
        result: ``(features, labels, material_ids, spacegroups,
            structures_path) -> the tuple the public function returns``.
    """

    log_prefix: str
    load: Callable[[Path], Optional[tuple]]
    compute: Callable[[list, Dict[str, Any]], Any]
    save: Callable[..., None]
    result: Callable[..., tuple]


def _soap_load(cache_path: Path) -> Optional[tuple]:
    cached = _load_cached_dataset(cache_path)
    if cached is None:
        return None
    X, labels, material_ids, spacegroups, mean, std = cached
    return (X, mean, std), labels, material_ids, spacegroups


def _soap_save(cache_path, features, labels, material_ids, spacegroups) -> None:
    X_std, feature_mean, feature_std = features
    _save_dataset_cache(
        cache_path, X_std, labels, material_ids, spacegroups, feature_mean, feature_std
    )


def _soap_result(features, labels, material_ids, spacegroups, structures_path):
    X_std, feature_mean, feature_std = features
    return (
        X_std,
        labels,
        material_ids,
        spacegroups,
        structures_path,
        feature_mean,
        feature_std,
    )


def _graph_load(cache_path: Path) -> Optional[tuple]:
    cached = _load_cached_graph_dataset(cache_path)
    if cached is None:
        return None
    graph_arrays, labels, material_ids, spacegroups = cached
    return graph_arrays, labels, material_ids, spacegroups


def _graph_save(cache_path, features, labels, material_ids, spacegroups) -> None:
    _save_graph_dataset_cache(cache_path, *features, labels, material_ids, spacegroups)


def _graph_result(features, labels, material_ids, spacegroups, structures_path):
    return tuple(features), labels, material_ids, spacegroups, structures_path


def _soap_featurizer() -> _Featurizer:
    return _Featurizer(
        log_prefix="Dataset",
        load=_soap_load,
        compute=lambda atoms, kw: _compute_soap_and_standardize(atoms, kw),
        save=_soap_save,
        result=_soap_result,
    )


def _graph_featurizer() -> _Featurizer:
    return _Featurizer(
        log_prefix="CGCNN dataset",
        load=_graph_load,
        compute=lambda atoms, kw: _compute_graphs(atoms, kw),
        save=_graph_save,
        result=_graph_result,
    )


def _mace_featurizer() -> _Featurizer:
    return _Featurizer(
        log_prefix="MACE dataset",
        load=_soap_load,  # same array schema as SOAP
        compute=lambda atoms, kw: _compute_mace_and_standardize(atoms, kw),
        save=_soap_save,
        result=_soap_result,
    )


@dataclasses.dataclass(frozen=True)
class _StructureSource:
    """Where a dataset's structures come from.

    Attributes:
        hit_detail: Tail of the cache-hit log line.
        miss_detail: Tail of the cache-miss log line.
        produce: ``augmentation -> (atoms_list, labels, material_ids,
            spacegroups)``, with augmentation already applied.
    """

    hit_detail: Tuple[str, tuple]
    miss_detail: Tuple[str, tuple]
    produce: Callable[[Optional[AugmentationConfig]], tuple]


def _fetch_source(
    crystal_systems: Sequence[str], limit_per_system: int, api_key: Optional[str]
) -> _StructureSource:
    def produce(augmentation):
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
                logger.info(
                    "Fetched %d structures for crystal_system=%s", n_fetched, cs
                )
            all_atoms.extend(atoms_list)
            labels.extend([cs.capitalize()] * len(atoms_list))

        if not all_atoms:
            raise ValueError("No structures fetched for the requested crystal systems.")

        material_ids = [a.info.get("material_id", "unknown") for a in all_atoms]
        spacegroups = [a.info.get("spacegroup", _UNKNOWN_SPACEGROUP) for a in all_atoms]
        return all_atoms, labels, material_ids, spacegroups

    systems = list(crystal_systems)
    return _StructureSource(
        hit_detail=("for crystal_systems=%s", (systems,)),
        miss_detail=("fetching structures for crystal_systems=%s", (systems,)),
        produce=produce,
    )


def _pyxtal_source(pyxtal_config: "PyxtalConfig", seed: int) -> _StructureSource:
    def produce(augmentation):
        # Lazy: dim_red.generate requires pyxtal, not a hard dim_red dependency.
        from dim_red.generate import GenerationConfig, generate_structures

        generation_kwargs = dataclasses.asdict(pyxtal_config)
        generation_kwargs.pop("seed", None)  # the resolved "seed" arg wins
        if generation_kwargs.get("candidate_num_ions") is None:
            generation_kwargs.pop("candidate_num_ions")
        all_atoms = generate_structures(
            GenerationConfig(seed=seed, **generation_kwargs)
        )

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
        return all_atoms, labels, material_ids, spacegroups

    return _StructureSource(
        hit_detail=("for pyxtal generation", ()),
        miss_detail=("generating structures with pyxtal", ()),
        produce=produce,
    )


def _get_or_build(
    featurizer: _Featurizer,
    source: _StructureSource,
    key: str,
    kwargs: Dict[str, Any],
    cache_dir: Union[str, Path],
    augmentation: Optional[AugmentationConfig],
) -> tuple:
    """The shared cache-or-build sequence behind every ``get_or_build_*``."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{key}.npz"
    structures_path = _structures_cache_path(cache_path)

    cached = featurizer.load(cache_path)
    if cached is not None:
        fmt, args = source.hit_detail
        logger.info(f"{featurizer.log_prefix} cache hit (%s) " + fmt, key, *args)
        features, labels, material_ids, spacegroups = cached
        return featurizer.result(
            features, labels, material_ids, spacegroups, structures_path
        )

    fmt, args = source.miss_detail
    logger.info(f"{featurizer.log_prefix} cache miss (%s); " + fmt, key, *args)

    all_atoms, labels, material_ids, spacegroups = source.produce(augmentation)
    _save_structures_cache(cache_path, all_atoms)
    features = featurizer.compute(all_atoms, kwargs)
    featurizer.save(cache_path, features, labels, material_ids, spacegroups)
    return featurizer.result(
        features, labels, material_ids, spacegroups, structures_path
    )


def resolve_augmentation(
    augmentation: Optional["PipelineAugmentationConfig"], default_seed: int
) -> Optional[AugmentationConfig]:
    """Converts the pipeline's own ``AugmentationConfig`` (see
    ``dim_red.pipeline.config``, independent of ``dim_red.augmentation``) into
    the real ``dim_red.augmentation.AugmentationConfig`` that the
    ``get_or_build_*`` builders actually use.
    ``None`` when augmentation is disabled; ``seed`` falls back to
    ``default_seed`` when ``augmentation.seed`` is ``None``."""
    if augmentation is None:
        return None
    seed = augmentation.seed if augmentation.seed is not None else default_seed
    return AugmentationConfig(
        n_augmented=augmentation.n_augmented,
        keep_original=augmentation.keep_original,
        jitter_probability=augmentation.jitter_probability,
        jitter_std=augmentation.jitter_std,
        vacancy_probability=augmentation.vacancy_probability,
        vacancy_atom_probability=augmentation.vacancy_atom_probability,
        max_vacancies=augmentation.max_vacancies,
        supercell_radius=augmentation.supercell_radius,
        seed=seed,
    )


def _resolve_augmentation(config: "RunConfig") -> Optional[AugmentationConfig]:
    """``resolve_augmentation`` applied to ``RunConfig.augmentation`` with
    ``RunConfig.seed`` as the fallback seed."""
    return resolve_augmentation(config.augmentation, config.seed)


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
    """The graph-based counterpart to ``get_or_build_dataset``: fetch
    structures and build CGCNN graph arrays, reusing a cached copy when
    available.

    Args:
        crystal_systems: Crystal systems to include.
        graph_kwargs: Keyword arguments for ``atoms_list_to_graph_arrays``
            (``dim_red.pipeline.config.GraphConfig.graph_kwargs()``).
        limit_per_system: Max structures fetched per crystal system.
        cache_dir: Directory where cached datasets live.
        api_key: Materials Project API key (falls back to ``MP_API_KEY``).
        augmentation: Optional ``dim_red.augmentation.AugmentationConfig``,
            applied before graph construction.

    Returns:
        ``(graph_arrays, labels, material_ids, spacegroups, structures_path)``
        where ``graph_arrays`` is the 5-tuple ``(local_species_idx, nbr_idx,
        nbr_fea, nbr_mask, atom_mask)`` -- 5 elements, not 7: no
        ``feature_mean``/``feature_std`` for graph features.
    """
    key = _graph_cache_key(
        crystal_systems, limit_per_system, graph_kwargs, augmentation
    )
    return _get_or_build(
        _graph_featurizer(),
        _fetch_source(crystal_systems, limit_per_system, api_key),
        key,
        graph_kwargs,
        cache_dir,
        augmentation,
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
    """The ``pyxtal`` counterpart to ``get_or_build_cgcnn_dataset``; same
    5-element return, with ``labels`` holding each structure's crystal
    family. ``seed`` is the *effective* generation seed (see
    ``get_or_build_pyxtal_dataset``).
    """
    key = _pyxtal_graph_cache_key(pyxtal_config, seed, graph_kwargs, augmentation)
    return _get_or_build(
        _graph_featurizer(),
        _pyxtal_source(pyxtal_config, seed),
        key,
        graph_kwargs,
        cache_dir,
        augmentation,
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
    called by ``dim_red.pipeline.single_run.run_single`` (cgcnn). Returns
    the 5-element graph-shaped tuple ``(graph_arrays, labels, material_ids,
    spacegroups, structures_path)``.
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


# --- MACE dataset caching (supcon_mace stacks, via dim_red.pipeline.stack_data)
#
# Unlike the CGCNN graph path above, MACE's body is frozen/pretrained -- there
# is no training loop to feed padded per-atom graph arrays into, so this
# caches the *final pooled per-structure embedding* directly, the same array
# schema as the SOAP path (_CACHE_ARRAY_KEYS), just produced by a frozen
# MaceEncoder forward pass instead of dscribe SOAP. The fetch/generate/
# augment/cache-raw-structures-to-<hash>.extxyz plumbing is reused unchanged
# once again (see module docstring) -- only feature computation and the cache
# key prefix ("mace-"/"mace-pyxtal-") differ. Reuses _CACHE_ARRAY_KEYS/
# _load_cached_dataset/_save_dataset_cache as-is (same schema as SOAP).


def _mace_cache_key(
    crystal_systems: Sequence[str],
    limit_per_system: int,
    mace_kwargs: Dict[str, Any],
    augmentation: Optional[AugmentationConfig] = None,
) -> str:
    """Stable hash identifying a MACE-featurized dataset built from (crystal
    systems, fetch limit, MACE hyperparameters, augmentation settings).
    Prefixed ``"mace-"`` so it can never collide with a SOAP/cgcnn cache hash
    even if it matched numerically.
    """
    payload = {
        "crystal_systems": sorted(cs.lower() for cs in crystal_systems),
        "limit_per_system": limit_per_system,
        "mace_kwargs": {k: mace_kwargs[k] for k in sorted(mace_kwargs)},
        "augmentation": _augmentation_payload(augmentation),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return "mace-" + hashlib.sha256(blob).hexdigest()[:16]


def _pyxtal_mace_cache_key(
    pyxtal_config: "PyxtalConfig",
    seed: int,
    mace_kwargs: Dict[str, Any],
    augmentation: Optional[AugmentationConfig] = None,
) -> str:
    """Stable hash identifying a MACE-featurized dataset built from (pyxtal
    generation config, seed, MACE hyperparameters, augmentation settings).
    Prefixed ``"mace-pyxtal-"``, mirrors ``_pyxtal_cache_key``.
    """
    payload = {
        "pyxtal": dataclasses.asdict(pyxtal_config),
        "seed": seed,
        "mace_kwargs": {k: mace_kwargs[k] for k in sorted(mace_kwargs)},
        "augmentation": _augmentation_payload(augmentation),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return "mace-pyxtal-" + hashlib.sha256(blob).hexdigest()[:16]


def _compute_mace_and_standardize(
    atoms_list: list, mace_kwargs: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns ``(X_std, feature_mean, feature_std)`` -- the MACE counterpart
    to ``_compute_soap_and_standardize``, used *instead of* it for
    ``supcon_mace`` stacks (a trained SupCon body, these are its raw *input*
    features, standardized the exact same way SOAP's raw input features are
    for plain ``supcon``). Runs a frozen, pretrained
    ``dim_red.mace.model.MaceEncoder`` forward pass over every structure
    (no training, no gradient), then standardizes the resulting embedding
    matrix the same way SOAP features are -- kept for consistency with
    ``dim_red.pipeline.benchmark``'s kNN/silhouette metrics and so newly
    applied structures can be standardized the same way without ever
    recomputing anything on the training set.

    ``mace_kwargs`` is imported lazily inside this function (not at module
    top) since ``dim_red.mace.model.MaceEncoder`` requires ``mace_jax``, not
    a hard ``dim_red`` dependency -- same convention
    ``get_or_build_pyxtal_dataset`` uses for ``dim_red.generate``/``pyxtal``.
    """
    from dim_red.mace.model import MaceEncoder

    logger.info("Computing MACE embeddings for %d structures", len(atoms_list))
    encoder = MaceEncoder(**mace_kwargs)
    X = np.asarray(encoder.encode(atoms_list), dtype=np.float64)
    mean, std = fit_standardization(X)
    X_std = apply_standardization(X, mean, std)
    return X_std.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def get_or_build_mace_dataset(
    crystal_systems: Sequence[str],
    mace_kwargs: Dict[str, Any],
    limit_per_system: int,
    cache_dir: Union[str, Path],
    api_key: Optional[str] = None,
    augmentation: Optional[AugmentationConfig] = None,
) -> Tuple[np.ndarray, List[str], List[str], List[int], Path, np.ndarray, np.ndarray]:
    """The MACE-embedding counterpart to ``get_or_build_dataset``: fetch
    structures, embed them with a frozen MACE model and standardize,
    reusing a cached copy when available. Same 7-element return and cache
    array schema as the SOAP path; only the featurization differs.

    Args:
        crystal_systems: Crystal systems to include.
        mace_kwargs: Keyword arguments for ``dim_red.mace.model.MaceEncoder``
            (``dim_red.pipeline.config.MaceConfig.mace_kwargs()``).
        limit_per_system: Max structures fetched per crystal system.
        cache_dir: Directory where cached datasets live.
        api_key: Materials Project API key (falls back to ``MP_API_KEY``).
        augmentation: Optional ``dim_red.augmentation.AugmentationConfig``.
    """
    key = _mace_cache_key(crystal_systems, limit_per_system, mace_kwargs, augmentation)
    return _get_or_build(
        _mace_featurizer(),
        _fetch_source(crystal_systems, limit_per_system, api_key),
        key,
        mace_kwargs,
        cache_dir,
        augmentation,
    )


def get_or_build_pyxtal_mace_dataset(
    pyxtal_config: "PyxtalConfig",
    seed: int,
    mace_kwargs: Dict[str, Any],
    cache_dir: Union[str, Path],
    augmentation: Optional[AugmentationConfig] = None,
) -> Tuple[np.ndarray, List[str], List[str], List[int], Path, np.ndarray, np.ndarray]:
    """The ``pyxtal`` counterpart to ``get_or_build_mace_dataset``; ``seed``
    is the *effective* generation seed (see ``get_or_build_pyxtal_dataset``).
    """
    key = _pyxtal_mace_cache_key(pyxtal_config, seed, mace_kwargs, augmentation)
    return _get_or_build(
        _mace_featurizer(),
        _pyxtal_source(pyxtal_config, seed),
        key,
        mace_kwargs,
        cache_dir,
        augmentation,
    )
