"""
Grid-sweep orchestration over VAE hidden-layer configurations (axis A) and
crystal-system subsets (axis B). Runs sharing a crystal-system subset reuse
the same dataset cache entry, so fetch + SOAP only run once per subset.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

from dim_red.pipeline.config import SweepConfig, expand_sweep
from dim_red.pipeline.single_run import run_single

logger = logging.getLogger("dim_red.pipeline")


def run_sweep(
    sweep: SweepConfig, cache_dir: Optional[Union[str, Path]] = None
) -> List[Path]:
    """Expand a SweepConfig's grid and run each combination sequentially.

    Returns:
        The list of run directories created, in grid order.
    """
    run_configs = expand_sweep(sweep)
    output_dir = Path(sweep.output_dir)
    resolved_cache_dir = Path(cache_dir) if cache_dir else output_dir / "_dataset_cache"

    logger.info(
        "Starting sweep: %d crystal-system set(s) x %d hidden-layer config(s) = %d run(s)",
        len(sweep.crystal_system_sets),
        len(sweep.hidden_layer_configs),
        len(run_configs),
    )

    run_dirs = []
    for i, config in enumerate(run_configs, start=1):
        logger.info(
            "[%d/%d] crystal_systems=%s hidden_dims=%s",
            i,
            len(run_configs),
            config.crystal_systems,
            config.vae.encoder_hidden_dim,
        )
        run_dirs.append(run_single(config, cache_dir=resolved_cache_dir))

    logger.info("Sweep complete: %d run(s) saved under %s", len(run_dirs), output_dir)
    return run_dirs
