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
directories in one call and produces one machine-readable table by default,
no plots. ``generate_benchmark_table(..., plot=True)`` (CLI: ``dimred-benchmark
... --plot``) additionally renders a small visual-comparison suite
(``generate_benchmark_plots``) -- box plots of the same numbers, pooled by
``model_kind`` rather than singling out one run, so this module still never
picks a "best" run/config, it only visualizes the spread.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

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
    ``dim_red.pipeline.compare.classification_accuracies_from_npz``), then to
    ``<run_dir>/tails/hierarchical/tail_predictions.npz`` (a hierarchical
    tail -- same ``family``/``spacegroup`` payload keys, see
    ``dim_red.pipeline.compare.hierarchical_accuracies_from_npz`` for its
    extra ``spacegroup_oracle`` number, not surfaced here). Only checks those
    exact paths, not a dedup-suffixed ``tails/classification-2``/
    ``tails/hierarchical-2`` from a repeated manual ``dimred-train-tail``
    call. Returns ``{}`` if none of these sources has classifier predictions.
    """
    accs = classification_accuracies_from_npz(run.embeddings)
    if accs:
        return accs
    for tail_kind in ("classification", "hierarchical"):
        tail_predictions = run.run_dir / "tails" / tail_kind / "tail_predictions.npz"
        if tail_predictions.exists():
            with np.load(tail_predictions) as npz:
                accs = classification_accuracies_from_npz(dict(npz.items()))
            if accs:
                return accs
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
    plot: bool = False,
    write_data_files: bool = False,
) -> Path:
    """Top-level entry point (also ``dimred-benchmark``): resolve ``inputs``
    (a mix of run directories and/or sweep directories) via
    ``collect_run_dirs``, build one ``benchmark_row`` per run, and write them
    as a single CSV to ``output_csv`` -- fieldnames are the union of every
    row's keys, in first-seen order, so a run missing some column (e.g. no
    spacegroup variety, no classifier) simply gets an empty cell there.

    If ``plot`` is True, also renders the ``generate_benchmark_plots`` suite
    into ``<output_csv's parent>/benchmark_plots`` from the same in-memory
    rows (``write_data_files`` forwarded to it, for whether each plot's data
    is additionally written to a CSV next to its PNG).
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

    if plot:
        generate_benchmark_plots(
            rows,
            output_csv.parent / "benchmark_plots",
            write_data_files=write_data_files,
        )

    return output_csv


# --- benchmark plots: same numbers as the table above, pooled by ------------
# --- model_kind into box plots (still never picking a "best" run) -----------


def _numeric_values_by_model_kind(
    rows: Sequence[Dict[str, Any]], key: str
) -> Dict[str, List[float]]:
    """Group ``rows[*][key]`` by ``rows[*]["model_kind"]``, parsing each value
    as a float and dropping missing/empty/NaN entries. Works uniformly
    whether ``rows`` are the float-valued dicts ``generate_benchmark_table``
    builds in memory, or string-valued rows re-read from an already-written
    benchmark CSV (e.g. via ``csv.DictReader``) -- a missing column reads
    back as ``""`` either way, which fails ``float()`` and is skipped.
    """
    grouped: Dict[str, List[float]] = {}
    for row in rows:
        try:
            value = float(row.get(key, ""))
        except (TypeError, ValueError):
            continue
        if math.isnan(value):
            continue
        grouped.setdefault(row["model_kind"], []).append(value)
    return grouped


def _model_kind_colors(model_kinds: Sequence[str]) -> Dict[str, Any]:
    """One stable ``tab10`` color per distinct ``model_kind``, assigned in
    sorted order so the same kind gets the same color across every plot in
    ``generate_benchmark_plots``' suite.
    """
    cmap = plt.get_cmap("tab10")
    return {kind: cmap(i % 10) for i, kind in enumerate(sorted(set(model_kinds)))}


def _add_model_kind_legend(fig: Any, colors: Dict[str, Any]) -> None:
    """Add one shared legend to ``fig`` mapping each ``model_kind`` to its
    box color (``colors``, from ``_model_kind_colors``), sorted by name.
    """
    handles = [
        Patch(facecolor=color, alpha=0.7, label=kind)
        for kind, color in sorted(colors.items())
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(handles), 6),
        bbox_to_anchor=(0.5, -0.05),
        fontsize=8,
        title="model_kind",
    )


def _boxplot_by_model_kind(
    ax: Any,
    grouped: Dict[str, List[float]],
    colors: Dict[str, Any],
    title: str,
    ylabel: str,
) -> bool:
    """Draw one box per ``model_kind`` in ``grouped`` (sorted by name) onto
    ``ax``, colored via ``colors``. Returns False (leaving ``ax`` untouched)
    if ``grouped`` is empty, so callers can hide/skip that subplot.
    """
    kinds = sorted(grouped)
    if not kinds:
        return False
    data = [grouped[k] for k in kinds]
    box = ax.boxplot(data, tick_labels=kinds, patch_artist=True, showmeans=True)
    for patch, kind in zip(box["boxes"], kinds):
        patch.set_facecolor(colors[kind])
        patch.set_alpha(0.7)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    return True


_ACCURACY_METRICS = ("family", "spacegroup")


def plot_classification_accuracy_by_model_kind(
    rows: Sequence[Dict[str, Any]],
    save_path: Optional[Union[str, Path]] = None,
    csv_path: Optional[Union[str, Path]] = None,
) -> None:
    """Box plot of family/spacegroup classification accuracy (``benchmark_row``'s
    ``"family"``/``"spacegroup"`` columns), one box per ``model_kind``, pooling
    every run of that kind found in ``rows`` -- shows the spread a
    hyperparameter sweep produced for that architecture rather than singling
    out any one run (see this module's docstring on why it never ranks). A
    ``model_kind`` with no classifier at all (e.g. a bare supcon body with no
    classification tail) simply gets no box. If no row has classification
    accuracy at all, no figure is produced.
    """
    grouped_by_metric = {
        metric: _numeric_values_by_model_kind(rows, metric)
        for metric in _ACCURACY_METRICS
    }
    active_metrics = [m for m in _ACCURACY_METRICS if grouped_by_metric[m]]
    if not active_metrics:
        logger.warning(
            "No run has classification accuracy; skipping accuracy-by-model_kind plot"
        )
        return

    if csv_path:
        csv_rows = [
            {"model_kind": kind, "metric": metric, "accuracy": value}
            for metric in active_metrics
            for kind, values in grouped_by_metric[metric].items()
            for value in values
        ]
        write_csv(csv_path, ["model_kind", "metric", "accuracy"], csv_rows)

    all_kinds = {k for metric in active_metrics for k in grouped_by_metric[metric]}
    colors = _model_kind_colors(all_kinds)

    fig, axes = plt.subplots(
        1, len(active_metrics), figsize=(5 * len(active_metrics), 5), squeeze=False
    )
    for ax, metric in zip(axes[0], active_metrics):
        _boxplot_by_model_kind(
            ax,
            grouped_by_metric[metric],
            colors,
            title=f"{metric.capitalize()} classification accuracy",
            ylabel="Accuracy",
        )
        ax.set_ylim(0, 1.05)
    fig.suptitle("Classification accuracy by model_kind")
    fig.tight_layout()
    _add_model_kind_legend(fig, colors)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        logger.info("Saved accuracy-by-model_kind plot to %s", save_path)
    plt.close(fig)


_2D_METRIC_SUFFIXES = ("silhouette", "kmeans_ari", "kmeans_nmi", "knn_accuracy")
_2D_LABEL_SETS = ("family_2d", "spacegroup_2d")


def plot_2d_quality_by_model_kind(
    rows: Sequence[Dict[str, Any]],
    save_path: Optional[Union[str, Path]] = None,
    csv_path: Optional[Union[str, Path]] = None,
) -> None:
    """Grid of box plots -- one row per active label set (``family_2d``, and
    ``spacegroup_2d`` when any run has usable spacegroup variety), one column
    per ``embedding_quality_metrics`` metric (silhouette/kmeans_ari/
    kmeans_nmi/knn_accuracy) -- showing how well families/phases separate in
    each run's 2D view (``_resolve_2d_embedding``, the same view a human
    would actually look at), grouped by ``model_kind``. A label set with no
    non-NaN value across every row is dropped entirely (its row of subplots
    isn't drawn); if neither label set has any data, no figure is produced.
    """
    grouped = {
        (label_set, suffix): _numeric_values_by_model_kind(
            rows, f"{label_set}_{suffix}"
        )
        for label_set in _2D_LABEL_SETS
        for suffix in _2D_METRIC_SUFFIXES
    }
    active_label_sets = [
        label_set
        for label_set in _2D_LABEL_SETS
        if any(grouped[(label_set, suffix)] for suffix in _2D_METRIC_SUFFIXES)
    ]
    if not active_label_sets:
        logger.warning(
            "No run has a usable 2D embedding-quality metric; "
            "skipping 2D-quality-by-model_kind plot"
        )
        return

    if csv_path:
        csv_rows = [
            {
                "model_kind": kind,
                "label_set": label_set,
                "metric": suffix,
                "value": value,
            }
            for label_set in active_label_sets
            for suffix in _2D_METRIC_SUFFIXES
            for kind, values in grouped[(label_set, suffix)].items()
            for value in values
        ]
        write_csv(csv_path, ["model_kind", "label_set", "metric", "value"], csv_rows)

    all_kinds = {
        kind
        for label_set in active_label_sets
        for suffix in _2D_METRIC_SUFFIXES
        for kind in grouped[(label_set, suffix)]
    }
    colors = _model_kind_colors(all_kinds)

    fig, axes = plt.subplots(
        len(active_label_sets),
        len(_2D_METRIC_SUFFIXES),
        figsize=(4.5 * len(_2D_METRIC_SUFFIXES), 4.5 * len(active_label_sets)),
        squeeze=False,
    )
    for row_idx, label_set in enumerate(active_label_sets):
        for col_idx, suffix in enumerate(_2D_METRIC_SUFFIXES):
            ax = axes[row_idx][col_idx]
            drawn = _boxplot_by_model_kind(
                ax,
                grouped[(label_set, suffix)],
                colors,
                title=f"{label_set}: {suffix}",
                ylabel=suffix,
            )
            if not drawn:
                ax.axis("off")
    fig.suptitle("2D embedding quality by model_kind")
    fig.tight_layout()
    _add_model_kind_legend(fig, colors)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        logger.info("Saved 2D-quality-by-model_kind plot to %s", save_path)
    plt.close(fig)


def generate_benchmark_plots(
    rows: Sequence[Dict[str, Any]],
    output_dir: Union[str, Path],
    write_data_files: bool = False,
) -> Path:
    """Render the benchmark visual-comparison suite
    (``plot_classification_accuracy_by_model_kind``,
    ``plot_2d_quality_by_model_kind``) into ``output_dir`` from ``rows`` --
    either ``generate_benchmark_table``'s own in-memory rows, or rows re-read
    from an already-written benchmark CSV (both accepted uniformly, see
    ``_numeric_values_by_model_kind``). Unlike ``dim_red.pipeline.compare``'s
    per-sweep PNGs (one point per run), these pool every run of a given
    ``model_kind`` into one box each, showing the spread a hyperparameter
    sweep produced for that architecture -- consistent with the rest of this
    module, it never ranks or picks a "best" run/config, only visualizes the
    numbers ``generate_benchmark_table`` already wrote.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _csv_path(name: str) -> Optional[Path]:
        return output_dir / name if write_data_files else None

    plot_classification_accuracy_by_model_kind(
        rows,
        save_path=output_dir / "accuracy_by_model_kind.png",
        csv_path=_csv_path("accuracy_by_model_kind.csv"),
    )
    plot_2d_quality_by_model_kind(
        rows,
        save_path=output_dir / "embedding_quality_2d_by_model_kind.png",
        csv_path=_csv_path("embedding_quality_2d_by_model_kind.csv"),
    )
    logger.info("Wrote benchmark plots to %s", output_dir)
    return output_dir
