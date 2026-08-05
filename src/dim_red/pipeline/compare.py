"""
Comparison suite for sweep output: discovers every run directory written by
``dim_red.pipeline.sweep.run_sweep`` under a sweep directory, and renders a
fixed set of PNG comparison plots -- overlaid loss curves, final-metric-vs-
hyperparameter plots for whichever hyperparameters actually varied across the
runs, a grid of latent-space scatter plots, and (when auxiliary heads were
used) a classification-accuracy comparison -- into ``<sweep_dir>/comparison/``.
Each ``plot_*`` function also accepts a ``csv_path`` to write the exact data
backing that plot, so ``generate_comparison_report`` produces a CSV alongside
every PNG (same basename, ``.csv`` instead of ``.png``).
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from dim_red.analysis.plotting import plot_spacegroup_histogram
from dim_red.pca import PCA
from dim_red.pipeline.config import (
    RunConfig,
    flatten_config_dict,
    load_run_config,
    run_config_to_dict,
)

logger = logging.getLogger("dim_red.pipeline")

# Crystal-system names abbreviated to their first few letters wherever they'd
# otherwise blow up axis labels/titles/legends (e.g. six systems joined by
# "-" easily exceeds a figure's width).
_CRYSTAL_SYSTEM_ABBREV_LEN = 3

# Config fields never worth treating as a "hyperparameter" to plot/label by:
# constant across a sweep's runs by construction (output_dir), unique per
# run rather than swept (name), or sensitive (fetch.api_key -- excluded so it
# can never end up in a plot title/filename even if it somehow varied).
_NON_HYPERPARAM_KEYS = {"fetch.api_key", "output_dir", "name"}


def _abbreviate_crystal_systems(systems: List[str]) -> str:
    return "-".join(sorted(s.lower()[:_CRYSTAL_SYSTEM_ABBREV_LEN] for s in systems))


def _hashable(value: Any) -> Any:
    """Make a config leaf value usable as a set/dict element: lists (e.g.
    ``crystal_systems``, ``encoder_hidden_dim``) become tuples, recursively.
    """
    if isinstance(value, list):
        return tuple(_hashable(v) for v in value)
    return value


def _write_csv(
    path: Union[str, Path], fieldnames: List[str], rows: List[Dict[str, Any]]
) -> None:
    """Write ``rows`` (a list of ``{fieldname: value}`` dicts) as CSV,
    the tidy/long-format data backing one of this module's comparison plots.
    """
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved plot data to %s", path)


def _format_hyperparam_value(path: str, value: Any) -> Any:
    """Render a flattened config value for display: abbreviate crystal
    systems, join other lists with "-", and pass numeric/scalar values
    through unchanged so numeric plotting/sorting still works.
    """
    if path == "fetch.crystal_systems":
        return _abbreviate_crystal_systems(value)
    if isinstance(value, list):
        return "-".join(str(v) for v in value)
    return value


# Short forms for the hyperparameter *names* themselves (as opposed to their
# values, handled by ``_format_hyperparam_value``) -- mirrors the prefixes
# ``dim_red.pipeline.single_run.make_run_name`` already uses for run
# directory names (hd, cs, aux, lf, lsg), extended to the rest of RunConfig
# so any swept field gets a short, plot/CSV-friendly key. Anything not listed
# here falls back to its unabbreviated config field name.
_HYPERPARAM_KEY_ABBREV = {
    "crystal_systems": "cs",
    "limit_per_system": "limit",
    "encoder_hidden_dim": "hd",
    "decoder_hidden_dim": "dhd",
    "latent_dim": "ld",
    "mirror": "mirror",
    "epochs": "ep",
    "batch_size": "bs",
    "learning_rate": "lr",
    "beta": "beta",
    "val_ratio": "val_ratio",
    "device": "device",
    "mode": "aux",
    "lambda_family": "lf",
    "lambda_spacegroup": "lsg",
    "head_hidden_dim": "hhd",
    "seed": "seed",
    "model": "model",
    "data_source": "src",
    "structures_per_family": "spf",
    "structures_per_spacegroup": "sps",
    "distribution": "dist",
    "n_species": "nsp",
    "patience": "pat",
    "min_delta": "mdelta",
    "restore_best_weights": "restore",
}


def _abbreviate_hyperparam_key(path: str) -> str:
    """Short form of a dotted hyperparameter path's leaf name, e.g.
    ``"vae.encoder_hidden_dim"`` -> ``"hd"``, for compact run labels.
    """
    key = path.rsplit(".", 1)[-1]
    return _HYPERPARAM_KEY_ABBREV.get(key, key)


@dataclass
class RunData:
    """Everything loaded from one run directory needed for comparison plots."""

    run_dir: Path
    config: RunConfig
    loss_history: Dict[str, np.ndarray]
    embeddings: Dict[str, np.ndarray]

    @property
    def label(self) -> str:
        """Full run directory name -- unique, used for identification only."""
        return self.run_dir.name

    @property
    def flat_config(self) -> Dict[str, Any]:
        """This run's config as ``{dotted_path: leaf_value}`` pairs."""
        return flatten_config_dict(run_config_to_dict(self.config))


def discover_runs(sweep_dir: Union[str, Path]) -> List[Path]:
    """Return every immediate subdirectory of ``sweep_dir`` that looks like a
    completed run (has both ``config.yaml`` and ``loss_history.csv``), sorted
    by name for reproducible plot ordering.
    """
    sweep_dir = Path(sweep_dir)
    run_dirs = sorted(
        p
        for p in sweep_dir.iterdir()
        if p.is_dir()
        and (p / "config.yaml").exists()
        and (p / "loss_history.csv").exists()
    )
    if not run_dirs:
        raise ValueError(f"No completed runs found directly under {sweep_dir}")
    return run_dirs


def _load_loss_history(path: Path) -> Dict[str, np.ndarray]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return {
        key: np.array([float(row[key]) for row in rows]) for key in reader.fieldnames
    }


def load_run(run_dir: Union[str, Path]) -> RunData:
    """Load one run directory's config, loss history and embeddings."""
    run_dir = Path(run_dir)
    config = load_run_config(run_dir / "config.yaml")
    loss_history = _load_loss_history(run_dir / "loss_history.csv")
    with np.load(run_dir / "embeddings.npz") as npz:
        embeddings = dict(npz.items())
    return RunData(
        run_dir=run_dir, config=config, loss_history=loss_history, embeddings=embeddings
    )


def load_runs(sweep_dir: Union[str, Path]) -> List[RunData]:
    return [load_run(d) for d in discover_runs(sweep_dir)]


def varying_hyperparams(runs: List[RunData]) -> List[str]:
    """Dotted-path config keys (see ``dim_red.pipeline.config.flatten_config_dict``)
    whose value differs across ``runs`` -- i.e. whatever was actually swept,
    for *any* config field, not a fixed set of named axes. Sorted for
    reproducible plot ordering.
    """
    flat_configs = [r.flat_config for r in runs]
    varying = []
    for key in flat_configs[0]:
        if key in _NON_HYPERPARAM_KEYS:
            continue
        values = {_hashable(fc.get(key)) for fc in flat_configs}
        if len(values) > 1:
            varying.append(key)
    return sorted(varying)


def run_labels(runs: List[RunData]) -> Dict[Path, str]:
    """Compact, plot-safe label per run, built from whichever hyperparameters
    actually vary across ``runs`` (e.g. ``"hd=128-64_lf=1"``, using the same
    short parameter-name abbreviations as run directory names -- see
    ``_HYPERPARAM_KEY_ABBREV``) -- so legends/titles/CSV rows reflect whatever
    was actually swept, for any parameter, instead of a fixed hidden-dims/
    crystal-systems assumption. Falls back to the run directory name if
    nothing varies (e.g. a single-run "sweep").
    """
    varying = varying_hyperparams(runs)
    if not varying:
        return {r.run_dir: r.label for r in runs}
    labels = {}
    for run in runs:
        flat = run.flat_config
        parts = [
            f"{_abbreviate_hyperparam_key(key)}={_format_hyperparam_value(key, flat[key])}"
            for key in varying
        ]
        labels[run.run_dir] = "_".join(parts)
    return labels


_LOSS_METRIC_ORDER = [
    "train_loss",
    "val_loss",
    "train_recon",
    "val_recon",
    "train_kl",
    "val_kl",
    "train_family_ce",
    "val_family_ce",
    "train_spacegroup_ce",
    "val_spacegroup_ce",
]


def available_loss_metrics(runs: List[RunData]) -> List[str]:
    """Loss-history columns present in every run's ``loss_history.csv``
    (excluding ``"epoch"``), e.g. ``val_loss`` plus its ``recon``/``kl``
    components and, when auxiliary heads were active, the family/spacegroup
    cross-entropy terms -- in a fixed, readable order (known metrics first,
    any unexpected extra column sorted after).
    """
    common = set(runs[0].loss_history)
    for run in runs[1:]:
        common &= set(run.loss_history)
    common.discard("epoch")
    ordered = [m for m in _LOSS_METRIC_ORDER if m in common]
    ordered += sorted(common.difference(ordered))
    return ordered


def plot_loss_curves(
    runs: List[RunData],
    metrics: List[str] = ("train_loss", "val_loss"),
    save_path: Optional[Union[str, Path]] = None,
    csv_path: Optional[Union[str, Path]] = None,
) -> None:
    """Overlay every run's loss curve, one subplot per metric in ``metrics``,
    one line per run.
    """
    labels = run_labels(runs)

    if csv_path:
        rows = [
            {
                "run": labels[run.run_dir],
                "epoch": int(epoch),
                "metric": metric,
                "value": float(value),
            }
            for run in runs
            for metric in metrics
            if metric in run.loss_history
            for epoch, value in zip(run.loss_history["epoch"], run.loss_history[metric])
        ]
        _write_csv(csv_path, ["run", "epoch", "metric", "value"], rows)

    fig, axes = plt.subplots(
        1, len(metrics), figsize=(6 * len(metrics), 5), squeeze=False
    )
    axes = axes[0]
    cmap = plt.get_cmap("tab10")

    for ax, metric in zip(axes, metrics):
        for i, run in enumerate(runs):
            if metric not in run.loss_history:
                continue
            values = run.loss_history[metric]
            ax.plot(
                run.loss_history["epoch"],
                values,
                label=labels[run.run_dir],
                color=cmap(i % 10),
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(True, linestyle="--", alpha=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=min(len(runs), 4), frameon=True
    )
    fig.suptitle("Loss curves across runs", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        logger.info("Saved loss-curve comparison to %s", save_path)
    plt.close(fig)


def final_metric_groups(
    runs: List[RunData], hyperparam: str, metric: str = "val_loss"
) -> Dict[Any, List[float]]:
    """Group each run's final (last-epoch) ``metric`` by its value of
    ``hyperparam`` -- any dotted-path config key, e.g. as returned by
    ``varying_hyperparams`` (``"vae.encoder_hidden_dim"``,
    ``"train.learning_rate"``, ``"aux_heads.lambda_family"``, ...).

    Multiple runs commonly share the same hyperparameter value -- e.g. a
    sweep's other axes vary underneath a fixed crystal-system set -- so each
    key maps to a *list* of values, one per run in that group, rather than a
    single aggregate.
    """
    groups: Dict[Any, List[float]] = {}
    for run in runs:
        flat = run.flat_config
        if hyperparam not in flat:
            raise ValueError(
                f"Unknown hyperparam {hyperparam!r}; not present in this run's "
                f"config (available: {sorted(flat)})"
            )
        key = _format_hyperparam_value(hyperparam, flat[hyperparam])
        groups.setdefault(key, []).append(run.loss_history[metric][-1])
    return groups


def plot_final_metric_vs_hyperparam(
    runs: List[RunData],
    hyperparam: str,
    metric: str = "val_loss",
    save_path: Optional[Union[str, Path]] = None,
    csv_path: Optional[Union[str, Path]] = None,
) -> None:
    """Plot each run's final (last-epoch) ``metric`` against its value of
    ``hyperparam``, aggregated (mean +/- std) over however many runs share
    that value, with the individual runs overlaid as jittered points so
    within-group spread stays visible. Every hyperparameter -- numeric or
    categorical -- uses the same mean +/- std point-and-line rendering;
    categorical values just sit at evenly spaced positions labeled with
    their (string) value instead of a numeric x-coordinate.
    """
    groups = final_metric_groups(runs, hyperparam, metric)
    is_numeric = all(isinstance(k, (int, float)) for k in groups)

    if csv_path:
        run_label_map = run_labels(runs)
        rows = [
            {
                "run": run_label_map[run.run_dir],
                hyperparam: _format_hyperparam_value(
                    hyperparam, run.flat_config[hyperparam]
                ),
                metric: float(run.loss_history[metric][-1]),
            }
            for run in runs
        ]
        _write_csv(csv_path, ["run", hyperparam, metric], rows)
    keys = sorted(groups) if is_numeric else sorted(groups, key=str)
    means = [float(np.mean(groups[k])) for k in keys]
    stds = [float(np.std(groups[k])) for k in keys]
    max_n = max(len(groups[k]) for k in keys)
    positions = keys if is_numeric else list(range(len(keys)))

    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(keys)), 5))
    jitter_rng = np.random.default_rng(0)

    for pos, k in zip(positions, keys):
        ys = groups[k]
        xs = pos + (
            jitter_rng.uniform(-0.15, 0.15, size=len(ys))
            if len(ys) > 1
            else np.zeros(1)
        )
        ax.scatter(xs, ys, color="tab:gray", alpha=0.5, s=25, zorder=2)

    ax.errorbar(
        positions,
        means,
        yerr=stds,
        fmt="o-",
        color="tab:blue",
        capsize=4,
        zorder=3,
        label="mean ± std" if max_n > 1 else "value",
    )

    if not is_numeric:
        ax.set_xticks(positions)
        ax.set_xticklabels([str(k) for k in keys])

    ax.set_xlabel(hyperparam)
    ax.set_ylabel(f"final {metric}")
    ax.set_title(
        f"Final {metric} vs {hyperparam} "
        f"({len(runs)} run(s) across {len(keys)} group(s))"
    )
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        logger.info(
            "Saved final-%s-vs-%s comparison to %s", metric, hyperparam, save_path
        )
    plt.close(fig)


def latent_grid_axes(runs: List[RunData]) -> Optional[tuple]:
    """The two varying hyperparameters with the most distinct values, so
    ``plot_latent_space_grid`` can lay itself out as a real (rows x cols)
    grid along those two swept axes -- e.g. hidden-dim depth x beta -- rather
    than an arbitrary flat sequence. ``None`` if fewer than 2 hyperparameters
    vary (falls back to a flat grid).
    """
    varying = varying_hyperparams(runs)
    if len(varying) < 2:
        return None
    counts = {
        key: len({_hashable(r.flat_config.get(key)) for r in runs}) for key in varying
    }
    ranked = sorted(varying, key=lambda k: (-counts[k], k))
    return ranked[0], ranked[1]


@dataclass(frozen=True)
class LatentUmapParams:
    """UMAP hyperparameters for projecting non-2D latent embeddings down to
    2D for ``plot_latent_space_grid`` (and, for consistency within one
    report, the same params also drive ``compute_embedding_baselines``'s
    UMAP baseline). Every field left ``None`` (the default) leaves that
    hyperparameter at ``umap-learn``'s own default -- see
    ``dim_red.umap.UMAP``.
    """

    n_neighbors: Optional[int] = None
    min_dist: Optional[float] = None
    metric: Optional[str] = None
    random_state: Optional[int] = None


def _make_umap(n_components: int, umap_params: Optional[LatentUmapParams]):
    """Build a ``dim_red.umap.UMAP`` instance from ``umap_params`` (``None``
    treated the same as an all-defaults ``LatentUmapParams()``). Imports
    ``dim_red.umap`` lazily so callers that never need it (e.g. every run's
    latent space already being 2D) don't require ``umap-learn`` installed.
    """
    from dim_red.umap import UMAP

    params = umap_params or LatentUmapParams()
    return UMAP(
        n_components=n_components,
        n_neighbors=params.n_neighbors,
        min_dist=params.min_dist,
        metric=params.metric,
        random_state=params.random_state,
    )


def compute_embedding_baselines(
    runs: List[RunData], umap_params: Optional[LatentUmapParams] = None
) -> Optional[Tuple[str, np.ndarray, Dict[str, np.ndarray]]]:
    """Fit classical baselines (PCA, and UMAP if ``umap-learn`` is installed),
    both to 2 components, on the first available run's saved standardized
    ``features`` -- the exact same input the model encoded -- as a reference
    to compare the learned latent space against. Every run sharing that
    dataset (same crystal systems/SOAP settings) would give an identical
    baseline, so only one is computed rather than one per run.

    Args:
        runs: Runs to search for a saved ``features`` array.
        umap_params: Hyperparameters for the UMAP baseline (see
            ``LatentUmapParams``); ``None`` uses ``umap-learn``'s defaults.

    Returns:
        ``(source_run_name, labels, {method_name: coords})``, or ``None``
        (with a warning) if no run has "features" saved -- e.g. it predates
        that artifact and needs to be rerun to get this comparison.
    """
    source = next((r for r in runs if "features" in r.embeddings), None)
    if source is None:
        logger.warning(
            "No run has saved 'features' (raw standardized descriptors); "
            "skipping PCA/UMAP baseline comparison -- rerun to get this."
        )
        return None

    X = source.embeddings["features"]
    point_labels = source.embeddings["labels"]
    baselines: Dict[str, np.ndarray] = {"PCA": PCA(n_components=2).fit_transform(X)}
    try:
        umap = _make_umap(n_components=2, umap_params=umap_params)
    except ImportError:
        logger.warning("umap-learn not installed; skipping UMAP baseline")
    else:
        baselines["UMAP"] = umap.fit_transform(X)
    return source.run_dir.name, point_labels, baselines


def plot_spacegroup_family_histogram(
    runs: List[RunData],
    save_path: Optional[Union[str, Path]] = None,
    csv_path: Optional[Union[str, Path]] = None,
) -> None:
    """Bar chart of spacegroup counts across the fetched dataset, colored by
    crystal family -- see ``dim_red.analysis.plotting.plot_spacegroup_histogram``.

    Computed once from the first run's saved ``spacegroups``/``labels``
    arrays (every run in ``run_single`` always saves these), same
    "compute once, from the first run" convention as
    ``compute_embedding_baselines``'s PCA/UMAP baseline: every run sharing
    that dataset (same crystal systems) would give an identical histogram.
    """
    source = runs[0]
    spacegroups = source.embeddings["spacegroups"].tolist()
    families = source.embeddings["labels"].tolist()

    if csv_path:
        rows = [
            {"spacegroup": sg, "family": family}
            for sg, family in zip(spacegroups, families)
        ]
        _write_csv(csv_path, ["spacegroup", "family"], rows)

    plot_spacegroup_histogram(
        spacegroups,
        families,
        title=f"Spacegroup distribution ({source.run_dir.name}'s dataset)",
        save_path=str(save_path) if save_path else None,
    )


def _project_run_to_2d(
    run: RunData, umap_params: Optional[LatentUmapParams]
) -> Tuple[Optional[np.ndarray], bool]:
    """This run's latent embeddings, as 2D coordinates ready to scatter-plot.

    Already-2D embeddings pass through unchanged. Anything else (1D, 3D+)
    is UMAP-projected down to 2D with ``umap_params`` -- so a run's own
    latent space stays representative (unlike naively plotting only its
    first two raw dimensions, which drops information and isn't a real
    projection at all).

    Returns:
        ``(coords, was_projected)``. ``coords`` is ``None`` (with a warning)
        if a projection was needed but ``umap-learn`` isn't installed --
        that run must then be dropped from the grid entirely, since there's
        no other way to make its non-2D embeddings plottable.
    """
    embeddings = run.embeddings["embeddings"]
    if embeddings.shape[1] == 2:
        return embeddings, False
    try:
        umap = _make_umap(n_components=2, umap_params=umap_params)
    except ImportError:
        logger.warning(
            "Run %s has %d-dimensional latent embeddings and umap-learn is "
            "not installed to project them down to 2D; skipping it from the "
            "latent-space grid.",
            run.run_dir.name,
            embeddings.shape[1],
        )
        return None, False
    return umap.fit_transform(embeddings), True


def plot_latent_space_grid(
    runs: List[RunData],
    save_path: Optional[Union[str, Path]] = None,
    csv_path: Optional[Union[str, Path]] = None,
    umap_params: Optional[LatentUmapParams] = None,
) -> None:
    """Grid of 2D latent-space scatter plots, colored by label. When at
    least 2 hyperparameters vary across ``runs``, the grid's rows/columns
    are laid out along the two with the most distinct values (e.g. hidden-dim
    depth x beta): the higher-cardinality one becomes columns and the other
    rows, so the figure comes out wider than it is tall rather than the
    reverse. A run's position reflects its actual hyperparameter values
    instead of an arbitrary flat sequence; falls back to a flat grid when
    fewer than 2 hyperparameters vary. If more than one run lands in the
    same (row, col) cell (a 3rd+ axis also varies), only the first (by run
    directory name) is shown there and the rest are dropped, with a warning.

    Runs whose latent space isn't 2D are UMAP-projected down to 2D for this
    plot (with ``umap_params``, see ``LatentUmapParams`` -- ``None`` uses
    ``umap-learn``'s own defaults) rather than skipped or truncated to their
    first two raw dimensions; such subplots are annotated "(UMAP)" so it's
    clear they show a projection, not the raw latent coordinates. A run is
    only actually dropped if it needs projecting and ``umap-learn`` isn't
    installed.

    An extra row is appended with PCA (and UMAP, if installed) fit to the
    same standardized features the model was trained on -- see
    ``compute_embedding_baselines`` -- so the learned latent space can be
    visually compared against classical dimensionality reduction. Skipped
    (with a warning) if no run has saved features (older runs).
    """
    plottable = []
    projected_coords: Dict[Path, np.ndarray] = {}
    was_projected: Dict[Path, bool] = {}
    for r in runs:
        coords, projected = _project_run_to_2d(r, umap_params)
        if coords is None:
            continue
        plottable.append(r)
        projected_coords[r.run_dir] = coords
        was_projected[r.run_dir] = projected
    if not plottable:
        logger.warning(
            "No runs could be shown in 2D (missing umap-learn for non-2D "
            "latent spaces); skipping latent-space grid"
        )
        return

    run_label_map = run_labels(plottable)
    cmap = plt.get_cmap("tab10")
    axes_keys = latent_grid_axes(plottable)

    if axes_keys is None:
        n_cols = min(3, len(plottable))
        n_rows = math.ceil(len(plottable) / n_cols)
        cells = {(i // n_cols, i % n_cols): [run] for i, run in enumerate(plottable)}
        row_titles = col_titles = None
    else:
        # latent_grid_axes returns (highest-cardinality, second) -- put the
        # highest-cardinality one in columns so the figure grows wider
        # rather than taller as the two axis sizes diverge.
        col_key, row_key = axes_keys
        row_seen: Dict[Any, Any] = {}
        col_seen: Dict[Any, Any] = {}
        for run in plottable:
            flat = run.flat_config
            row_seen.setdefault(_hashable(flat[row_key]), flat[row_key])
            col_seen.setdefault(_hashable(flat[col_key]), flat[col_key])
        row_order = sorted(row_seen)
        col_order = sorted(col_seen)
        row_titles = [_format_hyperparam_value(row_key, row_seen[k]) for k in row_order]
        col_titles = [_format_hyperparam_value(col_key, col_seen[k]) for k in col_order]
        row_index = {k: i for i, k in enumerate(row_order)}
        col_index = {k: i for i, k in enumerate(col_order)}

        cells = {}
        for run in sorted(plottable, key=lambda r: r.run_dir.name):
            flat = run.flat_config
            cell = (
                row_index[_hashable(flat[row_key])],
                col_index[_hashable(flat[col_key])],
            )
            cells.setdefault(cell, []).append(run)

        dropped = sum(len(rs) - 1 for rs in cells.values() if len(rs) > 1)
        if dropped:
            logger.warning(
                "%d run(s) shared a (%s, %s) grid cell with another run; only the "
                "first is shown in each such cell",
                dropped,
                row_key,
                col_key,
            )
        n_rows, n_cols = len(row_order), len(col_order)

    baselines_info = compute_embedding_baselines(plottable, umap_params=umap_params)
    n_baselines = len(baselines_info[2]) if baselines_info else 0
    grid_cols = max(n_cols, n_baselines) if baselines_info else n_cols
    total_rows = n_rows + (1 if baselines_info else 0)

    fig, axes = plt.subplots(
        total_rows, grid_cols, figsize=(5 * grid_cols, 4.5 * total_rows), squeeze=False
    )

    csv_rows: List[Dict[str, Any]] = []

    for (row, col), cell_runs in cells.items():
        ax = axes[row][col]
        run = cell_runs[0]
        coords = projected_coords[run.run_dir]
        projected = was_projected[run.run_dir]
        point_labels = run.embeddings["labels"]
        if csv_path:
            run_name = run_label_map[run.run_dir]
            csv_rows.extend(
                {
                    "source": run_name,
                    "row": row,
                    "col": col,
                    "is_baseline": False,
                    "projected_umap": projected,
                    "dim_0": float(coords[i, 0]),
                    "dim_1": float(coords[i, 1]),
                    "label": str(point_labels[i]),
                }
                for i in range(coords.shape[0])
            )
        for i, label in enumerate(sorted(set(point_labels.tolist()))):
            mask = point_labels == label
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                label=label,
                color=cmap(i % 10),
                alpha=0.8,
                edgecolors="w",
                s=40,
            )
        if row_titles is None:
            # Flat grid: row/col headers don't exist, so each subplot needs
            # its own label to identify which run it is.
            title = run_label_map[run.run_dir]
            if projected:
                title += " (UMAP)"
            ax.set_title(title, fontsize=8)
        elif projected:
            # Row/col grid: no per-subplot title slot, so a small corner
            # annotation is the only way to flag a UMAP-projected (non-2D
            # latent space) subplot instead of letting it look like raw
            # latent coordinates.
            ax.annotate(
                "UMAP",
                xy=(0.02, 0.98),
                xycoords="axes fraction",
                ha="left",
                va="top",
                fontsize=7,
                style="italic",
                alpha=0.6,
            )
        ax.legend(fontsize=6)
        ax.grid(True, linestyle="--", alpha=0.4)

    for row in range(n_rows):
        for col in range(grid_cols):
            if col >= n_cols or (row, col) not in cells:
                axes[row][col].axis("off")

    if baselines_info:
        source_name, base_labels, baselines = baselines_info
        baseline_row = n_rows
        for i, (method_name, coords) in enumerate(baselines.items()):
            ax = axes[baseline_row][i]
            if csv_path:
                csv_rows.extend(
                    {
                        "source": f"{method_name} baseline",
                        "row": baseline_row,
                        "col": i,
                        "is_baseline": True,
                        "projected_umap": method_name == "UMAP",
                        "dim_0": float(coords[j, 0]),
                        "dim_1": float(coords[j, 1]),
                        "label": str(base_labels[j]),
                    }
                    for j in range(coords.shape[0])
                )
            for j, label in enumerate(sorted(set(base_labels.tolist()))):
                mask = base_labels == label
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    label=label,
                    color=cmap(j % 10),
                    alpha=0.8,
                    edgecolors="w",
                    s=40,
                )
            ax.set_title(f"{method_name} baseline", fontsize=9, fontweight="bold")
            ax.legend(fontsize=6)
            ax.grid(True, linestyle="--", alpha=0.4)
        for col in range(n_baselines, grid_cols):
            axes[baseline_row][col].axis("off")

    if row_titles is not None:
        for row in range(n_rows):
            axes[row][0].set_ylabel(str(row_titles[row]), fontsize=9, fontweight="bold")
        for col in range(n_cols):
            axes[0][col].annotate(
                str(col_titles[col]),
                xy=(0.5, 1.12),
                xycoords="axes fraction",
                ha="center",
                fontsize=9,
                fontweight="bold",
            )
        suptitle = f"Latent space across runs (rows={row_key}, cols={col_key})"
    else:
        suptitle = "Latent space across runs"

    if baselines_info:
        suptitle += (
            f"\n(PCA/UMAP baseline computed from {baselines_info[0]}'s features)"
        )

    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if csv_path:
        _write_csv(
            csv_path,
            [
                "source",
                "row",
                "col",
                "is_baseline",
                "projected_umap",
                "dim_0",
                "dim_1",
                "label",
            ],
            csv_rows,
        )

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        logger.info("Saved latent-space grid comparison to %s", save_path)
    plt.close(fig)


def _aux_accuracies(run: RunData) -> Dict[str, float]:
    """Family/spacegroup classification accuracy for one run, using whichever
    auxiliary-head predictions and ground truth are present in its
    ``embeddings.npz``. Returns an empty dict if no aux heads were active.
    """
    emb = run.embeddings
    accs: Dict[str, float] = {}
    if "family_probs" in emb:
        pred = emb["family_classes"][emb["family_probs"].argmax(axis=1)]
        accs["family"] = float((pred == emb["labels"]).mean())
    if "spacegroup_probs" in emb and "spacegroups" in emb:
        pred = emb["spacegroup_classes"][emb["spacegroup_probs"].argmax(axis=1)]
        accs["spacegroup"] = float((pred == emb["spacegroups"]).mean())
    return accs


def plot_aux_accuracy_comparison(
    runs: List[RunData],
    save_path: Optional[Union[str, Path]] = None,
    csv_path: Optional[Union[str, Path]] = None,
) -> None:
    """Grouped bar chart of family/spacegroup classification accuracy across
    the runs that had auxiliary heads active. Runs without aux heads are
    skipped; if none had aux heads, no figure is produced.
    """
    run_label_map = run_labels(runs)
    per_run_accs = [(run_label_map[r.run_dir], _aux_accuracies(r)) for r in runs]
    per_run_accs = [(label, accs) for label, accs in per_run_accs if accs]
    if not per_run_accs:
        logger.warning(
            "No runs with auxiliary heads active; skipping accuracy comparison"
        )
        return

    if csv_path:
        rows = [
            {"run": label, "metric": metric, "accuracy": accuracy}
            for label, accs in per_run_accs
            for metric, accuracy in accs.items()
        ]
        _write_csv(csv_path, ["run", "metric", "accuracy"], rows)

    metric_names = sorted({m for _, accs in per_run_accs for m in accs})
    labels = [label for label, _ in per_run_accs]
    x = np.arange(len(labels))
    width = 0.8 / len(metric_names)

    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(labels)), 5))
    for i, metric in enumerate(metric_names):
        values = [accs.get(metric, 0.0) for _, accs in per_run_accs]
        ax.bar(x + i * width, values, width, label=f"{metric} accuracy")

    ax.set_xticks(x + width * (len(metric_names) - 1) / 2)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Auxiliary-head classification accuracy across runs")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        logger.info("Saved auxiliary-head accuracy comparison to %s", save_path)
    plt.close(fig)


def generate_comparison_report(
    sweep_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    write_data_files: bool = False,
    umap_params: Optional[LatentUmapParams] = None,
) -> Path:
    """Discover every run under ``sweep_dir`` and render the full comparison
    suite (loss curves -- one subplot per loss-history column common to every
    run, e.g. total/recon/KL and, when active, the aux-head cross-entropy or
    SupCon terms, not just the total -- final-metric-vs-hyperparameter plots
    for those same components against whatever hyperparameter varied, a
    spacegroup histogram colored by family, a latent-space grid (any non-2D
    run projected to 2D via UMAP, see ``plot_latent_space_grid``) with a
    PCA/UMAP baseline comparison, and an aux-heads accuracy comparison if
    applicable) into ``output_dir`` (default: ``<sweep_dir>/comparison``).

    Args:
        sweep_dir: Directory containing completed run subdirectories.
        output_dir: Where to write the comparison PNGs (and CSVs, if
            ``write_data_files``). Defaults to ``<sweep_dir>/comparison``.
        write_data_files: If True, also write each plot's underlying data to
            a CSV file of the same name (e.g. ``loss_curves.png`` /
            ``loss_curves.csv``), so the numbers behind a plot can be
            inspected or reprocessed without parsing the image. Off by
            default -- only the PNGs are written.
        umap_params: Hyperparameters (``LatentUmapParams``) for the UMAP
            projection used in the latent-space grid, both for non-2D runs
            and the UMAP baseline. ``None`` (default) uses ``umap-learn``'s
            own defaults.

    Returns:
        The directory the comparison PNGs (and CSVs) were written to.
    """
    sweep_dir = Path(sweep_dir)
    runs = load_runs(sweep_dir)
    output_dir = Path(output_dir) if output_dir else sweep_dir / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    def _csv_path(name: str) -> Optional[Path]:
        return output_dir / name if write_data_files else None

    logger.info("Comparing %d run(s) from %s", len(runs), sweep_dir)

    plot_loss_curves(
        runs,
        metrics=available_loss_metrics(runs),
        save_path=output_dir / "loss_curves.png",
        csv_path=_csv_path("loss_curves.csv"),
    )

    if len(runs) > 1:
        varying = varying_hyperparams(runs)
        for metric in available_loss_metrics(runs):
            for hyperparam in varying:
                safe_name = hyperparam.replace(".", "_")
                plot_final_metric_vs_hyperparam(
                    runs,
                    hyperparam,
                    metric=metric,
                    save_path=output_dir / f"final_{metric}_vs_{safe_name}.png",
                    csv_path=_csv_path(f"final_{metric}_vs_{safe_name}.csv"),
                )
    else:
        logger.info("Only one run found; skipping final-metric-vs-hyperparameter plots")

    plot_spacegroup_family_histogram(
        runs,
        save_path=output_dir / "spacegroup_histogram.png",
        csv_path=_csv_path("spacegroup_histogram.csv"),
    )
    plot_latent_space_grid(
        runs,
        save_path=output_dir / "latent_space_grid.png",
        csv_path=_csv_path("latent_space_grid.csv"),
        umap_params=umap_params,
    )
    plot_aux_accuracy_comparison(
        runs,
        save_path=output_dir / "aux_heads_accuracy.png",
        csv_path=_csv_path("aux_heads_accuracy.csv"),
    )

    logger.info("Comparison report saved to %s", output_dir)
    return output_dir
