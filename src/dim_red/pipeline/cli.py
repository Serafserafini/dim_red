"""
Command-line entrypoints for the dim_red pipeline (structures -> features ->
model), installed as console scripts by ``pip install -e .``:

    dimred-run configs/full_stack.example.yaml
    dimred-sweep configs/full_stack_sweep.example.yaml
    dimred-rerun runs/full-stack-example
    dimred-train-heads configs/train_heads.example.yaml runs/full-stack-example --heads-name wide
    dimred-compare runs/20261007-1
    dimred-apply new_structures.extxyz runs/full-stack-example
    dimred-benchmark runs/20261007-1 --output runs/benchmark.csv
    dimred-train-tail configs/tail_train_classification.example.yaml <cgcnn run dir>

Every command imports its own (possibly jax-pulling) dependencies lazily, so
e.g. ``dimred-compare`` never needs jax installed.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path
from typing import Optional


def _configure_console_logging() -> None:
    """Attach a console handler directly to the "dim_red.pipeline" logger.

    Configured directly on our own logger (with ``propagate=False``) rather
    than via ``logging.basicConfig``, so the output format and level don't
    depend on -- or get duplicated by -- whatever handlers a third-party
    import happened to attach to the root logger.
    """
    logger = logging.getLogger("dim_red.pipeline")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        logger.addHandler(handler)


def _is_full_stack_config(d: dict) -> bool:
    return any(key in d for key in ("model_kind", "family", "experts"))


def _do_run(config_path: str, cache_dir: Optional[str]) -> Path:
    from dim_red.pipeline.config import load_yaml

    _configure_console_logging()
    raw = load_yaml(config_path)
    cache = Path(cache_dir) if cache_dir else None
    if _is_full_stack_config(raw):
        from dim_red.pipeline.full_stack import DEFAULT_HEADS_NAME, FullStack
        from dim_red.pipeline.full_stack_config import full_stack_config_from_dict

        full_stack = FullStack.create(full_stack_config_from_dict(raw), cache_dir=cache)
        full_stack.fit_body()
        full_stack.fit_heads(DEFAULT_HEADS_NAME)
        run_dir = full_stack.run_dir
    else:
        from dim_red.pipeline.config import run_config_from_dict
        from dim_red.pipeline.single_run import run_single

        run_dir = run_single(run_config_from_dict(raw), cache_dir=cache)
    print(f"Run complete: {run_dir}")
    return run_dir


def _do_sweep(config_path: str, cache_dir: Optional[str]) -> list:
    from dim_red.pipeline.config import load_sweep_config
    from dim_red.pipeline.sweep import run_sweep

    _configure_console_logging()
    sweep_config = load_sweep_config(config_path)
    run_dirs = run_sweep(sweep_config, cache_dir=Path(cache_dir) if cache_dir else None)
    print(f"Sweep complete: {len(run_dirs)} run(s).")
    for run_dir in run_dirs:
        print(f" - {run_dir}")
    return run_dirs


def _do_rerun(run_dir: str, cache_dir: Optional[str]) -> Path:
    import yaml

    _configure_console_logging()
    run_dir = Path(run_dir)
    with open(run_dir / "config.yaml") as f:
        raw = yaml.safe_load(f)
    cache = Path(cache_dir) if cache_dir else None
    if "stacks" in raw:  # a FullStack run: config.yaml is the resolved config
        from dim_red.pipeline.full_stack import DEFAULT_HEADS_NAME, FullStack
        from dim_red.pipeline.full_stack_config import (
            full_stack_config_from_resolved_dict,
        )

        # FullStack.create allocates a fresh, deduplicated directory.
        full_stack = FullStack.create(
            full_stack_config_from_resolved_dict(raw), cache_dir=cache
        )
        full_stack.fit_body()
        full_stack.fit_heads(DEFAULT_HEADS_NAME)
        new_run_dir = full_stack.run_dir
    else:
        from dim_red.pipeline.config import load_run_config
        from dim_red.pipeline.single_run import run_single

        config = dataclasses.replace(
            load_run_config(run_dir / "config.yaml"), name=None
        )
        new_run_dir = run_single(config, cache_dir=cache)
    print(f"Rerun complete: {new_run_dir}")
    return new_run_dir


def _str_to_bool(value: str) -> bool:
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(
        f"Expected a boolean value (true/false), got {value!r}"
    )


def _do_compare(
    sweep_dir: str,
    output_dir: Optional[str],
    data_file: bool = False,
    umap_n_neighbors: Optional[int] = None,
    umap_min_dist: Optional[float] = None,
    umap_metric: Optional[str] = None,
    umap_random_state: Optional[int] = None,
    heads_name: Optional[str] = None,
) -> Path:
    from dim_red.pipeline.compare import LatentUmapParams, generate_comparison_report

    _configure_console_logging()
    umap_params = LatentUmapParams(
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric=umap_metric,
        random_state=umap_random_state,
    )
    report_dir = generate_comparison_report(
        sweep_dir,
        output_dir=output_dir,
        write_data_files=data_file,
        umap_params=umap_params,
        heads_name=heads_name,
    )
    print(f"Comparison report saved to {report_dir}")
    return report_dir


def _do_benchmark(
    inputs: list,
    output_csv: str,
    key_hyperparams: Optional[str],
    plot: bool = False,
    plot_data_file: bool = False,
    heads_name: Optional[str] = None,
) -> Path:
    from dim_red.pipeline.benchmark import (
        _DEFAULT_KEY_HYPERPARAMS,
        generate_benchmark_table,
    )

    _configure_console_logging()
    hyperparams = (
        [h.strip() for h in key_hyperparams.split(",")]
        if key_hyperparams
        else list(_DEFAULT_KEY_HYPERPARAMS)
    )
    table_path = generate_benchmark_table(
        inputs,
        output_csv,
        key_hyperparams=hyperparams,
        plot=plot,
        write_data_files=plot_data_file,
        heads_name=heads_name,
    )
    print(f"Benchmark table saved to {table_path}")
    if plot:
        print(f"Benchmark plots saved to {Path(output_csv).parent / 'benchmark_plots'}")
    return table_path


def _do_train_heads(
    config_path: str,
    run_dir: str,
    heads_name: str,
    stacks: Optional[list] = None,
    cache_dir: Optional[str] = None,
) -> Path:
    from dim_red.pipeline.config import load_yaml
    from dim_red.pipeline.full_stack import FullStack
    from dim_red.pipeline.full_stack_config import full_stack_config_from_dict

    _configure_console_logging()
    heads_config = full_stack_config_from_dict(load_yaml(config_path))
    full_stack = FullStack.open(
        run_dir, cache_dir=Path(cache_dir) if cache_dir else None
    )
    full_stack.fit_heads(heads_name, stacks=stacks, config=heads_config)
    print(f"Heads {heads_name!r} trained in {full_stack.run_dir}")
    return full_stack.run_dir


def train_heads_command(argv=None) -> None:
    """``dimred-train-heads <config> <run_dir> --heads-name NAME``: train a new
    set of classification + visualization heads on already-trained bodies."""
    parser = argparse.ArgumentParser(
        description="Train a named set of classification + visualization heads "
        "on the frozen bodies of an existing FullStack run. The config has the "
        "same schema as dimred-run's; only each listed stack's classifier/viz/"
        "seed blocks are read."
    )
    parser.add_argument("config", type=str, help="Path to a FullStack YAML config.")
    parser.add_argument("run_dir", type=str, help="Existing FullStack run directory.")
    parser.add_argument(
        "--heads-name", type=str, required=True, help="Name of the new heads set."
    )
    parser.add_argument(
        "--stacks",
        type=str,
        default=None,
        help="Comma-separated stacks to train (default: every stack in the config).",
    )
    args = parser.parse_args(argv)
    stacks = [s.strip() for s in args.stacks.split(",")] if args.stacks else None
    _do_train_heads(args.config, args.run_dir, args.heads_name, stacks)


def run_command(argv=None) -> None:
    """``dimred-run <config>``: run a single dim_red pipeline pass."""
    parser = argparse.ArgumentParser(description="Run a single dim_red pipeline pass.")
    parser.add_argument("config", type=str, help="Path to a single-run YAML config.")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Dataset cache directory (default: <output_dir>/_dataset_cache).",
    )
    args = parser.parse_args(argv)
    _do_run(args.config, args.cache_dir)


def sweep_command(argv=None) -> None:
    """``dimred-sweep <config>``: expand a sweep config's grid and run every combination."""
    parser = argparse.ArgumentParser(
        description="Run a dim_red hyperparameter sweep from a YAML config."
    )
    parser.add_argument("config", type=str, help="Path to a sweep YAML config.")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Dataset cache directory (default: <output_dir>/_dataset_cache).",
    )
    args = parser.parse_args(argv)
    _do_sweep(args.config, args.cache_dir)


def rerun_command(argv=None) -> None:
    """``dimred-rerun <run_dir>``: rerun an existing run from its saved config.yaml."""
    parser = argparse.ArgumentParser(
        description="Rerun an existing dim_red run directory from its saved config.yaml."
    )
    parser.add_argument(
        "run_dir", type=str, help="Path to an existing run directory to rerun."
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Dataset cache directory (default: <output_dir>/_dataset_cache).",
    )
    args = parser.parse_args(argv)
    _do_rerun(args.run_dir, args.cache_dir)


def _add_umap_args(parser: argparse.ArgumentParser) -> None:
    """Shared ``--umap-*`` flags for the latent-space grid's UMAP projection
    (applied to non-2D runs and the UMAP baseline) -- used by both
    ``compare_command`` and the flag-based ``_build_parser``. All default to
    ``None``, which leaves that hyperparameter at ``umap-learn``'s own
    default (see ``dim_red.pipeline.compare.LatentUmapParams``).
    """
    parser.add_argument(
        "--umap-n-neighbors",
        type=int,
        default=None,
        help="UMAP n_neighbors for the latent-space grid (default: umap-learn's own, 15).",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=None,
        help="UMAP min_dist for the latent-space grid (default: umap-learn's own, 0.1).",
    )
    parser.add_argument(
        "--umap-metric",
        type=str,
        default=None,
        help="UMAP metric for the latent-space grid (default: umap-learn's own, euclidean).",
    )
    parser.add_argument(
        "--umap-random-state",
        type=int,
        default=None,
        help="UMAP random_state for the latent-space grid (default: umap-learn's own, "
        "non-deterministic).",
    )


def compare_command(argv=None) -> None:
    """``dimred-compare <sweep_dir>``: render the comparison-plot suite for a sweep."""
    parser = argparse.ArgumentParser(
        description="Render loss-curve/hyperparameter/latent-space comparison "
        "plots for every run under a dim_red sweep directory."
    )
    parser.add_argument(
        "sweep_dir",
        type=str,
        help="Path to a sweep directory (e.g. runs/20260728-1).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to write the comparison PNGs (default: <sweep_dir>/comparison).",
    )
    parser.add_argument(
        "--data-file",
        type=_str_to_bool,
        default=False,
        metavar="{true,false}",
        help="Also write the data behind each comparison plot to a CSV file "
        "next to its PNG (default: false, PNGs only).",
    )
    _add_umap_args(parser)
    parser.add_argument(
        "--heads-name",
        type=str,
        default=None,
        help="Which named heads set to use (default: the default set).",
    )
    args = parser.parse_args(argv)
    _do_compare(
        args.sweep_dir,
        args.output_dir,
        args.data_file,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        umap_metric=args.umap_metric,
        umap_random_state=args.umap_random_state,
        heads_name=args.heads_name,
    )


def benchmark_command(argv=None) -> None:
    """``dimred-benchmark <input>... --output <csv>``: assemble one wide CSV
    comparing every run found across the given run/sweep directories --
    possibly spanning different ``model_kind``s -- on standardized,
    dimensionality-agnostic embedding-quality and classification metrics.
    """
    parser = argparse.ArgumentParser(
        description="Assemble a benchmark CSV (one row per run and stack) from "
        "one or more dim_red run/sweep directories."
    )
    parser.add_argument(
        "inputs",
        type=str,
        nargs="+",
        help="One or more run directories and/or sweep directories (e.g. "
        "runs/tuning_supcon/20260728-1) -- a sweep directory is expanded to "
        "every completed run found directly under it.",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to write the benchmark CSV to."
    )
    parser.add_argument(
        "--key-hyperparams",
        type=str,
        default=None,
        help="Comma-separated dotted config paths to include as columns "
        "(default: model.encoder_hidden_dim,model.latent_dim,"
        "model.body_train.learning_rate,model.body_train.batch_size,"
        "model.body_train.epochs,seed).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also render a visual-comparison suite (box plots of "
        "classification accuracy and 2D embedding-quality metrics, pooled "
        "by model_kind) into <output>'s sibling benchmark_plots/ directory.",
    )
    parser.add_argument(
        "--plot-data-file",
        type=_str_to_bool,
        default=False,
        metavar="{true,false}",
        help="With --plot: also write the data behind each plot to a CSV "
        "file next to its PNG (default: false, PNGs only).",
    )
    parser.add_argument(
        "--heads-name",
        type=str,
        default=None,
        help="Which named heads set to use (default: the default set).",
    )
    args = parser.parse_args(argv)
    _do_benchmark(
        args.inputs,
        args.output,
        args.key_hyperparams,
        plot=args.plot,
        plot_data_file=args.plot_data_file,
        heads_name=args.heads_name,
    )


def _do_apply(
    structures_path: str,
    run_dir: str,
    output_dir: Optional[str],
    label_field: Optional[str],
    umap_n_neighbors: Optional[int] = None,
    umap_min_dist: Optional[float] = None,
    umap_metric: Optional[str] = None,
    umap_random_state: Optional[int] = None,
    heads_name: Optional[str] = None,
) -> Path:
    from dim_red.pipeline.run_layout import is_full_stack_run

    _configure_console_logging()
    if is_full_stack_run(run_dir):
        from dim_red.pipeline.full_stack import apply_to_structures

        result_dir = apply_to_structures(
            run_dir, structures_path, output_dir=output_dir, heads_name=heads_name
        )
        print(f"Applied structures saved to {result_dir}")
        return result_dir

    from dim_red.pipeline.compare import LatentUmapParams
    from dim_red.pipeline.inference import apply_model_to_structures

    umap_params = LatentUmapParams(
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric=umap_metric,
        random_state=umap_random_state,
    )
    result_dir = apply_model_to_structures(
        run_dir,
        structures_path,
        output_dir=output_dir,
        label_field=label_field,
        umap_params=umap_params,
    )
    print(f"Applied structures saved to {result_dir}")
    return result_dir


def apply_command(argv=None) -> None:
    """``dimred-apply <structures> <run_dir>``: apply an already-trained run's
    model to new structures and plot them in its latent space alongside the
    original training dataset.
    """
    parser = argparse.ArgumentParser(
        description="Apply an already-trained dim_red run's model to new "
        "structures, and plot them in its latent space alongside the "
        "original training dataset."
    )
    parser.add_argument(
        "structures",
        type=str,
        help="Path to an extended-XYZ file with the structures to apply the model to.",
    )
    parser.add_argument(
        "run_dir",
        type=str,
        help="Path to a completed run directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to write the applied embeddings/plot (default: <run_dir>/applied).",
    )
    parser.add_argument(
        "--label-field",
        type=str,
        default=None,
        help="An atoms.info key to color/label the new structures by (default: "
        "a single 'Applied structure' group).",
    )
    _add_umap_args(parser)
    parser.add_argument(
        "--heads-name",
        type=str,
        default=None,
        help="Which named heads set to use (default: the default set).",
    )
    args = parser.parse_args(argv)
    _do_apply(
        args.structures,
        args.run_dir,
        args.output_dir,
        args.label_field,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        umap_metric=args.umap_metric,
        umap_random_state=args.umap_random_state,
        heads_name=args.heads_name,
    )


def _do_train_tail(config_path: str, run_dir: str) -> Path:
    import dataclasses

    from dim_red.pipeline.config import load_tail_train_config
    from dim_red.pipeline.tail_training import train_tail

    _configure_console_logging()
    tail_config = load_tail_train_config(config_path)
    # The run directory is always supplied here (not read from the YAML),
    # so the same tail-training config can be reused across many runs
    # without editing it each time -- overrides whatever run_dir (if any)
    # the YAML itself set.
    tail_config = dataclasses.replace(tail_config, run_dir=run_dir)
    tail_dir = train_tail(tail_config)
    print(f"Tail training complete: {tail_dir}")
    return tail_dir


def train_tail_command(argv=None) -> None:
    """``dimred-train-tail <config> <run_dir>``: cgcnn-only -- freeze an
    already-trained cgcnn run's body and train a classification or
    visualization tail on top of it. See
    ``configs/tail_train_classification.example.yaml``/
    ``configs/tail_train_visualization.example.yaml``.
    """
    parser = argparse.ArgumentParser(
        description="Freeze an already-trained cgcnn run's body and train a "
        "classification or visualization tail on top of it."
    )
    parser.add_argument("config", type=str, help="Path to a tail-training YAML config.")
    parser.add_argument(
        "run_dir",
        type=str,
        help="Path to the completed run directory whose frozen body to "
        "attach the tail to (overrides run_dir in the config, if it sets "
        "one).",
    )
    args = parser.parse_args(argv)
    _do_train_tail(args.config, args.run_dir)
