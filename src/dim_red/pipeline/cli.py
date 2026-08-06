"""
Command-line entrypoints for the unified fetch -> SOAP -> VAE pipeline.

Preferred usage (console scripts installed by ``pip install -e .``):
    dimred-run configs/single_run.example.yaml
    dimred-sweep configs/sweep.example.yaml
    dimred-rerun runs/20260728-1/hd-128_cs-cubic
    dimred-compare runs/20260728-1
    dimred-apply new_structures.extxyz runs/20260728-1/hd-128_cs-cubic

Equivalent, flag-based form (``python -m``), kept for scripting/backward
compatibility:
    python -m dim_red.pipeline.cli --config configs/single_run.example.yaml
    python -m dim_red.pipeline.cli --config configs/sweep.example.yaml --sweep
    python -m dim_red.pipeline.cli --rerun runs/20260728-1/hd-128_cs-cubic
    python -m dim_red.pipeline.cli --compare runs/20260728-1
    python -m dim_red.pipeline.cli --apply new_structures.extxyz --apply-run runs/20260728-1/hd-128_cs-cubic

Note: no single-letter flags are defined here on purpose. ``dim_red.vae.training``
imports ``learned_optimization``, which parses ``sys.argv`` with ``absl`` at
import time and raises on ambiguous short flags such as ``-v``. For the same
reason, every command below imports its own (possibly jax-pulling)
dependencies lazily, so e.g. ``dimred-compare`` never needs jax installed.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path
from typing import Optional


def _configure_console_logging() -> None:
    """Attach a console handler directly to the "dim_red.pipeline" logger.

    Not using ``logging.basicConfig`` here: importing ``run_single``/``run_sweep``
    (which pull in ``dim_red.vae.training`` -> ``learned_optimization`` -> absl)
    already attaches a handler to the root logger as an import side effect, so
    ``basicConfig`` would be a no-op and its ``level=INFO`` would be silently
    ignored. Configuring our own logger directly, with ``propagate=False``,
    sidesteps that and also avoids duplicate/differently-formatted output from
    absl's root handler.
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


def _do_run(config_path: str, cache_dir: Optional[str]) -> Path:
    from dim_red.pipeline.config import load_run_config
    from dim_red.pipeline.single_run import run_single

    _configure_console_logging()
    run_config = load_run_config(config_path)
    run_dir = run_single(run_config, cache_dir=Path(cache_dir) if cache_dir else None)
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
    from dim_red.pipeline.config import load_run_config
    from dim_red.pipeline.single_run import run_single

    _configure_console_logging()
    run_dir = Path(run_dir)
    config = load_run_config(run_dir / "config.yaml")
    # Force a fresh (deduplicated) directory rather than overwriting the original run.
    config = dataclasses.replace(config, name=None)
    new_run_dir = run_single(config, cache_dir=Path(cache_dir) if cache_dir else None)
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
    )
    print(f"Comparison report saved to {report_dir}")
    return report_dir


def run_command(argv=None) -> None:
    """``dimred-run <config>``: run a single fetch -> SOAP -> VAE pass."""
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
    args = parser.parse_args(argv)
    _do_compare(
        args.sweep_dir,
        args.output_dir,
        args.data_file,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        umap_metric=args.umap_metric,
        umap_random_state=args.umap_random_state,
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
) -> Path:
    from dim_red.pipeline.compare import LatentUmapParams
    from dim_red.pipeline.inference import apply_model_to_structures

    _configure_console_logging()
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
        help="Path to a completed run directory (needs config.yaml, "
        "dataset.extxyz, model_params.msgpack and embeddings.npz).",
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
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified fetch -> SOAP -> VAE training pipeline for dim_red."
    )
    parser.add_argument(
        "--config", type=str, help="Path to a single-run or sweep YAML config."
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Treat --config as a sweep config and expand its grid into multiple runs.",
    )
    parser.add_argument(
        "--rerun",
        type=str,
        default=None,
        help="Path to an existing run directory; reruns it from its saved config.yaml.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Dataset cache directory (default: <output_dir>/_dataset_cache).",
    )
    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        help=(
            "Path to a sweep directory (e.g. runs/20260728-1); renders the "
            "comparison-plot suite for every run found under it into "
            "<sweep_dir>/comparison/ instead of running anything."
        ),
    )
    parser.add_argument(
        "--data-file",
        type=_str_to_bool,
        default=False,
        metavar="{true,false}",
        help="With --compare: also write the data behind each comparison "
        "plot to a CSV file next to its PNG (default: false, PNGs only).",
    )
    parser.add_argument(
        "--apply",
        type=str,
        default=None,
        help=(
            "Path to an extended-XYZ file with structures to apply an "
            "already-trained run's model to; requires --apply-run. Plots "
            "them in that run's latent space alongside its original "
            "training dataset instead of running anything."
        ),
    )
    parser.add_argument(
        "--apply-run",
        type=str,
        default=None,
        help="With --apply: path to the completed run directory whose model to apply.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="With --apply: where to write the applied embeddings/plot "
        "(default: <apply-run>/applied).",
    )
    parser.add_argument(
        "--label-field",
        type=str,
        default=None,
        help="With --apply: an atoms.info key to color/label the new "
        "structures by (default: a single 'Applied structure' group).",
    )
    _add_umap_args(parser)
    return parser


def main(argv=None) -> None:
    """Flag-based entrypoint for ``python -m dim_red.pipeline.cli``; prefer the
    dedicated ``dimred-run``/``dimred-sweep``/``dimred-rerun``/``dimred-compare``/
    ``dimred-apply`` console scripts for interactive use.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.compare:
        _do_compare(
            args.compare,
            None,
            args.data_file,
            umap_n_neighbors=args.umap_n_neighbors,
            umap_min_dist=args.umap_min_dist,
            umap_metric=args.umap_metric,
            umap_random_state=args.umap_random_state,
        )
        return

    if args.apply:
        if not args.apply_run:
            parser.error("--apply requires --apply-run.")
        _do_apply(
            args.apply,
            args.apply_run,
            args.output_dir,
            args.label_field,
            umap_n_neighbors=args.umap_n_neighbors,
            umap_min_dist=args.umap_min_dist,
            umap_metric=args.umap_metric,
            umap_random_state=args.umap_random_state,
        )
        return

    if args.rerun:
        _do_rerun(args.rerun, args.cache_dir)
        return

    if not args.config:
        parser.error(
            "--config is required unless --rerun, --compare or --apply is used."
        )

    if args.sweep:
        _do_sweep(args.config, args.cache_dir)
    else:
        _do_run(args.config, args.cache_dir)


if __name__ == "__main__":
    main()
