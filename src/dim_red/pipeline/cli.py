"""
Command-line entrypoints for the unified fetch -> SOAP -> VAE pipeline.

Preferred usage (console scripts installed by ``pip install -e .``):
    dimred-run configs/single_run.example.yaml
    dimred-sweep configs/sweep.example.yaml
    dimred-rerun runs/20260728-1/hd-128_cs-cubic
    dimred-compare runs/20260728-1

Equivalent, flag-based form (``python -m``), kept for scripting/backward
compatibility:
    python -m dim_red.pipeline.cli --config configs/single_run.example.yaml
    python -m dim_red.pipeline.cli --config configs/sweep.example.yaml --sweep
    python -m dim_red.pipeline.cli --rerun runs/20260728-1/hd-128_cs-cubic
    python -m dim_red.pipeline.cli --compare runs/20260728-1

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


def _do_compare(sweep_dir: str, output_dir: Optional[str]) -> Path:
    from dim_red.pipeline.compare import generate_comparison_report

    _configure_console_logging()
    report_dir = generate_comparison_report(sweep_dir, output_dir=output_dir)
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
    args = parser.parse_args(argv)
    _do_compare(args.sweep_dir, args.output_dir)


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
    return parser


def main(argv=None) -> None:
    """Flag-based entrypoint for ``python -m dim_red.pipeline.cli``; prefer the
    dedicated ``dimred-run``/``dimred-sweep``/``dimred-rerun``/``dimred-compare``
    console scripts for interactive use.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.compare:
        _do_compare(args.compare, None)
        return

    if args.rerun:
        _do_rerun(args.rerun, args.cache_dir)
        return

    if not args.config:
        parser.error("--config is required unless --rerun or --compare is used.")

    if args.sweep:
        _do_sweep(args.config, args.cache_dir)
    else:
        _do_run(args.config, args.cache_dir)


if __name__ == "__main__":
    main()
