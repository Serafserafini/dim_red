"""Small helpers shared by ``single_run`` and ``tail_training``."""

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _make_unique_run_dir(output_dir: Path, name: str) -> Path:
    """Create and return ``output_dir / name``, deduplicating with a
    ``-<n>`` suffix if that directory already exists (e.g. re-running the
    same unnamed config into the same ``output_dir``) so runs never silently
    overwrite one another now that names carry no timestamp.
    """
    run_dir = output_dir / name
    suffix = 1
    while True:
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            suffix += 1
            run_dir = output_dir / f"{name}-{suffix}"


def _split_indices(n_samples: int, val_ratio: float, seed: int) -> tuple:
    """Reproducible train/val index split (at least one sample per side)."""
    n_val = max(1, int(round(n_samples * val_ratio)))
    n_val = min(n_val, n_samples - 1)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    return indices[n_val:], indices[:n_val]


def _split_indices_grouped(material_ids: List, val_ratio: float, seed: int) -> tuple:
    """Reproducible train/val split at the *group* level (unique
    ``material_id``), so every augmented copy of a structure lands in the
    same split as its original and its sibling copies.

    ``dim_red.augmentation.augment_structures`` keeps ``material_id``
    identical across an original structure and every one of its augmented
    (jittered/vacancy) copies (only ``source_material_id`` changes) --
    splitting per-row instead of per-group therefore routinely put a
    near-duplicate of a training structure into validation, inflating val
    accuracy (a val row is very often just a jittered/vacancy twin of a
    structure whose other copies are being trained on). Splitting at the
    ``material_id`` level closes that leak: every copy of a given structure
    goes to the same side.

    When every ``material_id`` is already unique (no augmentation, or
    augmentation that never duplicates a ``material_id``), this degenerates
    to exactly one group per sample, so it falls back to ``_split_indices``
    and reproduces its split byte-for-byte -- every pre-existing,
    non-augmented config's split is therefore completely unaffected by this
    function's introduction; only augmented datasets, which never had a
    guarantee like this before, see a different (larger, since it's no
    longer optimistic) split.
    """
    n_samples = len(material_ids)
    material_ids_arr = np.asarray(material_ids)
    unique_groups, inverse = np.unique(material_ids_arr, return_inverse=True)
    n_groups = unique_groups.shape[0]
    if n_groups == n_samples:
        return _split_indices(n_samples, val_ratio, seed)

    n_val_groups = max(1, int(round(n_groups * val_ratio)))
    n_val_groups = min(n_val_groups, n_groups - 1)
    rng = np.random.default_rng(seed)
    shuffled_group_order = rng.permutation(n_groups)
    val_group_ids = shuffled_group_order[:n_val_groups]
    val_mask = np.isin(inverse, val_group_ids)
    return np.flatnonzero(~val_mask), np.flatnonzero(val_mask)


def _build_vocab_ids(values: List) -> Tuple[List, np.ndarray]:
    """Map arbitrary hashable values to a sorted vocabulary and integer ids.

    Returns:
        ``(vocab, ids)`` where ``vocab[i]`` is the value for class id ``i``.
    """
    vocab = sorted(set(values))
    value_to_id = {v: i for i, v in enumerate(vocab)}
    ids = np.array([value_to_id[v] for v in values], dtype=np.int64)
    return vocab, ids


def _save_loss_history(path: Path, history: Dict[str, List[float]]) -> None:
    fieldnames = ["epoch"] + list(history.keys())
    n_epochs = len(next(iter(history.values())))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(n_epochs):
            row = {"epoch": epoch + 1}
            row.update({k: v[epoch] for k, v in history.items()})
            writer.writerow(row)
