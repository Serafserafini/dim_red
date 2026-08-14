"""
Cross-architecture benchmark table: given one or more run/sweep directories
(possibly spanning different ``model_kind``s), assembles a single wide CSV
with one row per run, quantifying quality for the user to read rather than
picking a "best" run automatically -- this module never ranks or selects.
Each row leads with its two headline quality signals: family classification
accuracy (``run_classification_accuracies``), then how well families/phases
separate specifically in the 2D view a human would actually look at
(``embedding_quality_metrics`` on ``_resolve_2d_embedding(run)``, prefixed
``family_2d``/``spacegroup_2d``). Everything else -- key hyperparameters,
final loss-history values (reference only, NOT comparable across
``model_kind``s -- see ``dim_red.pipeline.compare``'s module docstring on why
loss columns differ by kind), and the same quality metrics on the run's
*native*-dimensionality embedding -- follows for reference. Unlike
``dim_red.pipeline.compare`` (one sweep directory, a fixed suite of PNGs),
this module accepts any mix of individual run directories and sweep
directories in one call and produces one machine-readable table, no plots.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from dim_red.analysis.metrics import embedding_quality_metrics
from dim_red.pca import PCA
from dim_red.pipeline.compare import (
    RunData,
    classification_accuracies_from_npz,
    discover_runs,
    load_run,
    write_csv,
)

logger = logging.getLogger("dim_red.pipeline")

_DEFAULT_KEY_HYPERPARAMS = (
    "model",
    "vae.latent_dim",
    "vae.encoder_hidden_dim",
    "train.learning_rate",
    "train.batch_size",
    "train.epochs",
    "seed",
)

_LOG_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S,%f"

# Every key embedding_quality_metrics({"family_2d": ..., "spacegroup_2d": ...})
# produces -- written explicitly as NaN when _resolve_2d_embedding can't
# produce a 2D view at all, so this quality-signal column group is always
# present (same rationale as embedding_quality_metrics's own per-label-set
# NaN behavior: every row keeps the same key set).
_2D_METRIC_KEYS = tuple(
    f"{name}_{suffix}"
    for name in ("family_2d", "spacegroup_2d")
    for suffix in ("silhouette", "kmeans_ari", "kmeans_nmi", "knn_accuracy")
)


def _resolve_2d_embedding(run: RunData) -> Optional[np.ndarray]:
    """The most faithful 2D representation available for ``run`` -- the same
    view a human would actually look at in a plot -- in preference order:

    1. An already-trained ``viz_dim: 2`` visualization tail's own embeddings
       (``<run_dir>/tails/visualization/tail_embeddings.npz``): a genuine
       learned 2D projection (SupCon loss on the frozen body, see
       ``dim_red.supcon.tail_training``), not a post-hoc reduction.
    2. The run's native ``embeddings`` array, if it's already 2D.
    3. Otherwise a deterministic PCA projection to 2 components
       (``dim_red.pca.PCA``, the same one ``dim_red.pipeline.compare`` uses
       for its classical baseline) -- no extra dependency on ``umap-learn``.

    ``None`` if the native embedding has fewer than 2 dimensions (nothing
    meaningful to reduce).
    """
    tail_path = run.run_dir / "tails" / "visualization" / "tail_embeddings.npz"
    if tail_path.exists():
        with np.load(tail_path) as npz:
            tail_embeddings = npz["embeddings"]
            if tail_embeddings.shape[1] == 2:
                return tail_embeddings

    embeddings = run.embeddings["embeddings"]
    if embeddings.shape[1] == 2:
        return embeddings
    if embeddings.shape[1] < 2:
        return None
    return PCA(n_components=2).fit_transform(embeddings)


def collect_run_dirs(inputs: Sequence[Union[str, Path]]) -> List[Path]:
    """Resolve a mix of individual run directories and sweep directories into
    a flat, deduplicated, sorted list of run directories. Each ``inputs``
    entry that is itself a completed run (has ``config.yaml`` and
    ``loss_history.csv`` directly inside it) is included as-is; otherwise
    it's treated as a sweep directory and expanded via
    ``dim_red.pipeline.compare.discover_runs``.
    """
    resolved: List[Path] = []
    for raw in inputs:
        path = Path(raw)
        if (path / "config.yaml").exists() and (path / "loss_history.csv").exists():
            resolved.append(path.resolve())
        else:
            resolved.extend(p.resolve() for p in discover_runs(path))
    seen = set()
    unique: List[Path] = []
    for path in resolved:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return sorted(unique)


def run_classification_accuracies(run: RunData) -> Dict[str, float]:
    """Family/spacegroup classification accuracy for ``run``, uniformly
    across storage locations: tries ``run.embeddings`` first (covers
    vae/autoencoder/cgcnn's built-in heads); if that has no classifier
    predictions (e.g. a supcon body, or any run without auxiliary heads),
    falls back to ``<run_dir>/tails/classification/tail_predictions.npz`` if
    present (a supcon or cgcnn classification tail -- same payload keys, see
    ``dim_red.pipeline.compare.classification_accuracies_from_npz``). Only
    checks that exact path, not a dedup-suffixed ``tails/classification-2``
    from a repeated manual ``dimred-train-tail`` call. Returns ``{}`` if
    neither source has classifier predictions.
    """
    accs = classification_accuracies_from_npz(run.embeddings)
    if accs:
        return accs
    tail_predictions = run.run_dir / "tails" / "classification" / "tail_predictions.npz"
    if tail_predictions.exists():
        with np.load(tail_predictions) as npz:
            return classification_accuracies_from_npz(dict(npz.items()))
    return {}


def _log_span_seconds(log_path: Path) -> Optional[float]:
    """Wall-clock seconds between the first and last parsable timestamp in a
    ``run.log``-style file (``"%(asctime)s [%(levelname)s] %(message)s"``,
    written by every ``dim_red.pipeline`` entry point). ``None`` if the file
    doesn't exist or has no parsable line.
    """
    if not log_path.exists():
        return None
    first_ts = last_ts = None
    with open(log_path) as f:
        for line in f:
            try:
                ts = datetime.strptime(line.split(" [", 1)[0], _LOG_TIMESTAMP_FORMAT)
            except ValueError:
                continue
            if first_ts is None:
                first_ts = ts
            last_ts = ts
    if first_ts is None:
        return None
    return (last_ts - first_ts).total_seconds()


def run_wall_clock_seconds(run_dir: Path) -> Optional[float]:
    """Wall-clock training time for ``run_dir``'s phase-1 body training,
    from ``run.log``'s own timestamps. When ``RunConfig.tails`` auto-trains a
    tail right after phase 1, its ``FileHandler`` stays attached alongside
    the parent run's, so the parent ``run.log`` also captures the tail's log
    lines -- each ``tails/<kind>/run.log``'s own span is subtracted back out
    here so this reports phase-1 time specifically, comparable across a run
    with vs. without auto-tails. ``None`` if ``run.log`` is missing or has no
    parsable timestamp.
    """
    total = _log_span_seconds(run_dir / "run.log")
    if total is None:
        return None
    tails_dir = run_dir / "tails"
    if tails_dir.is_dir():
        for tail_log in tails_dir.glob("*/run.log"):
            nested = _log_span_seconds(tail_log)
            if nested is not None:
                total -= nested
    return max(total, 0.0)


def _csv_safe(value: Any) -> Any:
    """Join list-valued config leaves (e.g. ``encoder_hidden_dim``) into a
    single CSV-cell-friendly string; pass everything else through unchanged.
    """
    if isinstance(value, list):
        return "-".join(str(v) for v in value)
    return value


def benchmark_row(
    run: RunData,
    key_hyperparams: Sequence[str] = _DEFAULT_KEY_HYPERPARAMS,
) -> Dict[str, Any]:
    """One benchmark-table row for ``run``, its two headline quality signals
    first (in priority order), everything else after for reference:

    1. **Primary** -- ``run_classification_accuracies`` (``"family"``, and
       ``"spacegroup"`` when available): crystal-family classification
       accuracy.
    2. **Secondary** -- ``embedding_quality_metrics`` computed on
       ``_resolve_2d_embedding(run)`` (keys prefixed ``family_2d``/
       ``spacegroup_2d``): how well families/phases separate specifically
       in the 2D view a human would actually look at, as opposed to
       whatever the native ``latent_dim`` happens to be. Written as NaN
       (``_2D_METRIC_KEYS``) when no 2D view can be produced at all (native
       embedding has fewer than 2 dimensions), so the column group is
       always present.

    Then: identifying fields (``run_dir``/``model_kind``/``n_samples``/
    ``wall_clock_seconds``), the requested ``key_hyperparams`` (any dotted
    path into ``run.flat_config``; missing ones are written as ``""``),
    each loss-history column's final (last-epoch) value prefixed ``final_``
    (reference only -- not comparable across ``model_kind``s, see this
    module's docstring), and every ``embedding_quality_metrics`` key for
    the run's *native* ``embeddings`` (present unconditionally for every
    ``model_kind``, regardless of ``latent_dim``).
    """
    flat = run.flat_config
    row: Dict[str, Any] = {}
    row.update(run_classification_accuracies(run))

    embedding_2d = _resolve_2d_embedding(run)
    if embedding_2d is not None:
        row.update(
            embedding_quality_metrics(
                embedding_2d,
                {
                    "family_2d": run.embeddings["labels"],
                    "spacegroup_2d": run.embeddings["spacegroups"],
                },
            )
        )
    else:
        row.update({key: float("nan") for key in _2D_METRIC_KEYS})

    row["run_dir"] = str(run.run_dir)
    row["model_kind"] = run.config.model_kind
    row["n_samples"] = int(run.embeddings["embeddings"].shape[0])
    row["wall_clock_seconds"] = run_wall_clock_seconds(run.run_dir)
    for key in key_hyperparams:
        row[key] = _csv_safe(flat.get(key, ""))
    for metric, values in run.loss_history.items():
        if metric == "epoch":
            continue
        row[f"final_{metric}"] = float(values[-1]) if len(values) else float("nan")
    row.update(
        embedding_quality_metrics(
            run.embeddings["embeddings"],
            {
                "family": run.embeddings["labels"],
                "spacegroup": run.embeddings["spacegroups"],
            },
        )
    )
    return row


def generate_benchmark_table(
    inputs: Sequence[Union[str, Path]],
    output_csv: Union[str, Path],
    key_hyperparams: Sequence[str] = _DEFAULT_KEY_HYPERPARAMS,
) -> Path:
    """Top-level entry point (also ``dimred-benchmark``): resolve ``inputs``
    (a mix of run directories and/or sweep directories) via
    ``collect_run_dirs``, build one ``benchmark_row`` per run, and write them
    as a single CSV to ``output_csv`` -- fieldnames are the union of every
    row's keys, in first-seen order, so a run missing some column (e.g. no
    spacegroup variety, no classifier) simply gets an empty cell there.
    """
    run_dirs = collect_run_dirs(inputs)
    rows = [benchmark_row(load_run(d), key_hyperparams) for d in run_dirs]

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    output_csv = Path(output_csv)
    write_csv(output_csv, fieldnames, rows)
    logger.info("Wrote benchmark table (%d run(s)) to %s", len(rows), output_csv)
    return output_csv
