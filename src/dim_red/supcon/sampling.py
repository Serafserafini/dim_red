"""
Balanced batch sampler for SupCon training: an alternative to plain random
shuffling that guarantees every batch contains multiple examples per crystal
family (and, optionally, per spacegroup within each chosen family), so the
SupCon loss (``dim_red.supcon.training.supcon_loss``) reliably has positives
available for every anchor -- useful for datasets less balanced than the
pyxtal-generated one this package was originally built against.

Does not replace ``dim_red.supcon.training._iter_batches`` (the existing
random-shuffle path stays untouched and remains the default); this module is
purely additive, selected via ``BatchingConfig.strategy == "balanced"`` (see
``dim_red.pipeline.config``).

Logs onto the same ``"dim_red.pipeline"`` logger name every other pipeline
module uses (not a ``dim_red.supcon``-prefixed one) purely so P/S-clamping
warnings raised here land in a run's ``run.log`` -- this is a plain
``logging`` name, not an import, so it doesn't create a dependency on
``dim_red.pipeline``.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger("dim_red.pipeline")


def _sample_indices(pool: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n`` indices from ``pool``: without replacement when the pool is
    large enough, with replacement only when necessary (``len(pool) < n``).
    """
    replace = len(pool) < n
    return rng.choice(pool, size=n, replace=replace)


def balanced_batch_indices(
    family_ids: np.ndarray,
    spacegroup_ids: Optional[np.ndarray],
    P: Optional[int],
    K: int,
    S: Optional[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Build the row indices for one balanced batch.

    Chooses ``P`` families (all families present when ``P`` is ``None``,
    clamped down with a warning if ``P`` exceeds what's available), then for
    each chosen family either:

    - samples ``K`` rows directly from that family (``S`` is ``None``), or
    - chooses ``S`` spacegroups within that family (clamped down with a
      warning if ``S`` exceeds the spacegroups actually present for that
      family) and splits ``K`` across them as evenly as possible
      (``K // S_eff``, with the remainder ``K % S_eff`` going to the first
      few chosen spacegroups), sampling that many rows from each.

    Args:
        family_ids: Integer family id per row, shape ``(n_rows,)``.
        spacegroup_ids: Integer spacegroup id per row, shape ``(n_rows,)``.
            Required (not ``None``) when ``S`` is not ``None``.
        P: Number of families to include, or ``None`` for all present.
        K: Number of examples per chosen family.
        S: Number of spacegroups to stratify by within each chosen family,
            or ``None`` to skip spacegroup stratification.
        rng: Numpy random generator, reused across batches/epochs by the
            caller for a single reproducible stream.

    Returns:
        A 1D array of row indices, length ``P_eff * K`` where ``P_eff`` is
        ``P`` clamped to the number of families actually present (can only
        be smaller than the requested ``P * K``, never larger).
    """
    available_families = np.unique(family_ids)
    n_available = len(available_families)
    if P is None:
        P_eff = n_available
    else:
        P_eff = min(P, n_available)
        if P > n_available:
            logger.warning(
                "batching.balanced_params.P=%d exceeds %d families present in "
                "this split; clamping to %d",
                P,
                n_available,
                n_available,
            )
    chosen_families = rng.choice(available_families, size=P_eff, replace=False)

    batch_index_chunks: List[np.ndarray] = []
    for family in chosen_families:
        family_pool = np.flatnonzero(family_ids == family)

        if S is None:
            batch_index_chunks.append(_sample_indices(family_pool, K, rng))
            continue

        family_spacegroups = np.unique(spacegroup_ids[family_pool])
        n_sg_available = len(family_spacegroups)
        S_eff = min(S, n_sg_available)
        if S > n_sg_available:
            logger.warning(
                "batching.balanced_params.S=%d exceeds %d spacegroups present "
                "for family %s; clamping to %d",
                S,
                n_sg_available,
                family,
                n_sg_available,
            )
        if S_eff > K:
            logger.warning(
                "batching.balanced_params.S=%d > K=%d for family %s; only %d "
                "of the %d chosen spacegroups will get any sample this batch",
                S_eff,
                K,
                family,
                K,
                S_eff,
            )
        chosen_spacegroups = rng.choice(family_spacegroups, size=S_eff, replace=False)

        base, remainder = divmod(K, S_eff)
        for i, spacegroup in enumerate(chosen_spacegroups):
            count = base + (1 if i < remainder else 0)
            if count == 0:
                continue
            spacegroup_pool = family_pool[spacegroup_ids[family_pool] == spacegroup]
            batch_index_chunks.append(_sample_indices(spacegroup_pool, count, rng))

    indices = np.concatenate(batch_index_chunks)
    rng.shuffle(indices)
    return indices


def iter_balanced_batches(
    arrays: Tuple[np.ndarray, ...],
    family_ids: np.ndarray,
    spacegroup_ids: Optional[np.ndarray],
    P: Optional[int],
    K: int,
    S: Optional[int],
    n_batches: int,
    rng: np.random.Generator,
) -> List[Tuple[np.ndarray, ...]]:
    """Build ``n_batches`` balanced batches, slicing ``arrays`` by each
    batch's indices (see ``balanced_batch_indices``).

    Same return shape as ``dim_red.supcon.training._iter_batches`` --
    ``[(arrays[0][idx], arrays[1][idx], ...), ...]`` -- so a caller's epoch
    loop doesn't need to change beyond picking which of the two functions to
    call.

    Args:
        arrays: The arrays to slice per batch (typically ``(X, family_np,
            spacegroup_np)``), all the same length as ``family_ids``.
        family_ids: Integer family id per row, used only to choose indices
            (independent of whether ``arrays`` includes a family array at
            all, or includes one with different values -- e.g. a dummy
            all-zeros array when the SupCon family loss term is inactive but
            balanced batching is still requested).
        spacegroup_ids: Integer spacegroup id per row, or ``None`` if ``S``
            is ``None``. Same independence from ``arrays`` as ``family_ids``.
        P: Number of families per batch (``None`` -> all present).
        K: Number of examples per family per batch.
        S: Number of spacegroups per family per batch (``None`` -> family-only).
        n_batches: How many batches to build.
        rng: Numpy random generator, shared across batches for a single
            reproducible stream.
    """
    batches = []
    for _ in range(n_batches):
        idx = balanced_batch_indices(family_ids, spacegroup_ids, P, K, S, rng)
        batches.append(tuple(a[idx] for a in arrays))
    return batches
