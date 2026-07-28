"""
Caching layer for the fetch -> SOAP -> standardize dataset-construction step,
so that a hidden-layer sweep (axis A) does not repeat the Materials Project
fetch and SOAP computation for every combination that shares the same
crystal-system subset (axis B).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from dim_red.fetch import fetch_structures_by_crystal_system
from dim_red.soap import compute_soap
from dim_red.utils import standardize

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


def get_or_build_dataset(
    crystal_systems: Sequence[str],
    soap_kwargs: Dict[str, Any],
    limit_per_system: int,
    cache_dir: Union[str, Path],
    api_key: Optional[str] = None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Fetch structures, compute a global SOAP descriptor per structure, and
    standardize the result -- reusing a cached copy on disk when available.

    Args:
        crystal_systems: Crystal systems to include (as accepted by
            ``fetch_structures_by_crystal_system``).
        soap_kwargs: Keyword arguments for ``compute_soap`` minus ``average``
            (the pipeline always requests ``average="outer"``).
        limit_per_system: Max structures fetched per crystal system.
        cache_dir: Directory where cached datasets (``<hash>.npz``) live.
        api_key: Materials Project API key (falls back to ``MP_API_KEY``).

    Returns:
        ``(X_std, labels, material_ids)`` where ``X_std`` has shape
        ``(n_samples, n_features)``.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(crystal_systems, limit_per_system, soap_kwargs)
    cache_path = cache_dir / f"{key}.npz"

    if cache_path.exists():
        logger.info(
            "Dataset cache hit (%s) for crystal_systems=%s", key, list(crystal_systems)
        )
        cached = np.load(cache_path)
        return cached["X"], cached["labels"].tolist(), cached["material_ids"].tolist()

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

    effective_soap_kwargs = dict(soap_kwargs)
    if effective_soap_kwargs.get("species") is None:
        effective_soap_kwargs["species"] = sorted(
            {sym for a in all_atoms for sym in a.get_chemical_symbols()}
        )
    effective_soap_kwargs["average"] = "outer"

    logger.info("Computing SOAP descriptors for %d structures", len(all_atoms))
    soap_vectors = compute_soap(all_atoms, **effective_soap_kwargs)
    X = np.asarray(soap_vectors).reshape(len(all_atoms), -1)
    X_std = standardize(X)

    np.savez(
        cache_path,
        X=X_std,
        labels=np.array(labels),
        material_ids=np.array(material_ids),
    )
    logger.info("Dataset cached at %s (shape=%s)", cache_path, X_std.shape)

    return X_std, labels, material_ids
