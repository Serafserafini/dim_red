"""
Command-line entrypoint for the unified fetch -> SOAP -> VAE pipeline.

Usage:
    python -m dim_red.pipeline.cli --config configs/single_run.example.yaml
    python -m dim_red.pipeline.cli --config configs/sweep.example.yaml --sweep
    python -m dim_red.pipeline.cli --rerun runs/20260728-153000-123456_hd-128-cs-cubic

Note: no single-letter flags are defined here on purpose. ``dim_red.vae.training``
imports ``learned_optimization``, which parses ``sys.argv`` with ``absl`` at
import time and raises on ambiguous short flags such as ``-v``.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

from dim_red.pipeline.config import load_run_config, load_sweep_config
from dim_red.pipeline.single_run import run_single
from dim_red.pipeline.sweep import run_sweep


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
    return parser


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


def main(argv=None) -> None:
    _configure_console_logging()
    parser = _build_parser()
    args = parser.parse_args(argv)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    if args.rerun:
        run_dir = Path(args.rerun)
        config = load_run_config(run_dir / "config.yaml")
        # Force a fresh timestamped directory rather than overwriting the original run.
        config = dataclasses.replace(config, name=None)
        new_run_dir = run_single(config, cache_dir=cache_dir)
        print(f"Rerun complete: {new_run_dir}")
        return

    if not args.config:
        parser.error("--config is required unless --rerun is used.")

    if args.sweep:
        sweep_config = load_sweep_config(args.config)
        run_dirs = run_sweep(sweep_config, cache_dir=cache_dir)
        print(f"Sweep complete: {len(run_dirs)} run(s).")
        for run_dir in run_dirs:
            print(f" - {run_dir}")
    else:
        run_config = load_run_config(args.config)
        run_dir = run_single(run_config, cache_dir=cache_dir)
        print(f"Run complete: {run_dir}")


if __name__ == "__main__":
    main()
