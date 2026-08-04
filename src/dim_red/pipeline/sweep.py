"""
Grid-sweep orchestration: expands a ``SweepConfig``'s grid (the Cartesian
product of however many dotted-path axes it declares -- any RunConfig field
can be swept) into one run per combination. Runs sharing the same crystal
systems/SOAP settings/fetch limit reuse the same dataset cache entry, so
fetch + SOAP only run once per distinct combination of those.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

from dim_red.pipeline.config import SweepConfig, expand_sweep
from dim_red.pipeline.single_run import run_single

logger = logging.getLogger("dim_red.pipeline")

_SWEEP_DIR_RE = re.compile(r"^\d{8}-(\d+)$")


def _next_sweep_dir(output_dir: Path) -> Path:
    """Allocate this sweep invocation's directory, named ``<date>-<n>`` where
    ``n`` is one more than the highest sibling number already present under
    ``output_dir`` (regardless of that sibling's own date), so concurrent
    sweeps never collide and the individual run folders inside can be named
    by hyperparameters alone.
    """
    today = datetime.now().strftime("%Y%m%d")
    last_n = 0
    if output_dir.exists():
        for p in output_dir.iterdir():
            if not p.is_dir():
                continue
            m = _SWEEP_DIR_RE.match(p.name)
            if m:
                last_n = max(last_n, int(m.group(1)))
    sweep_dir = output_dir / f"{today}-{last_n + 1}"
    sweep_dir.mkdir(parents=True, exist_ok=False)
    return sweep_dir


def run_sweep(
    sweep: SweepConfig, cache_dir: Optional[Union[str, Path]] = None
) -> List[Path]:
    """Expand a SweepConfig's grid and run each combination sequentially.

    Each invocation gets its own ``<output_dir>/<date>-<n>`` directory (see
    ``_next_sweep_dir``); individual run folders inside it are named from
    their swept hyperparameters only (see ``make_run_name``). The dataset
    cache stays shared at ``<output_dir>/_dataset_cache`` across sweep
    invocations, since it doesn't depend on which sweep produced it.

    Returns:
        The list of run directories created, in grid order.
    """
    run_configs = expand_sweep(sweep)
    output_dir = Path(sweep.output_dir)
    resolved_cache_dir = Path(cache_dir) if cache_dir else output_dir / "_dataset_cache"
    sweep_dir = _next_sweep_dir(output_dir)

    logger.info(
        "Starting sweep: %d axis/axes (%s) = %d run(s), saved under %s",
        len(sweep.grid),
        sorted(sweep.grid),
        len(run_configs),
        sweep_dir,
    )

    run_dirs = []
    for i, config in enumerate(run_configs, start=1):
        data_scope = (
            config.fetch.crystal_systems
            if config.data_source == "fetch"
            else config.pyxtal
        )
        logger.info(
            "[%d/%d] data_source=%s scope=%s hidden_dims=%s",
            i,
            len(run_configs),
            config.data_source,
            data_scope,
            config.vae.encoder_hidden_dim,
        )
        config = dataclasses.replace(config, output_dir=str(sweep_dir))
        run_dirs.append(run_single(config, cache_dir=resolved_cache_dir))

    logger.info("Sweep complete: %d run(s) saved under %s", len(run_dirs), sweep_dir)
    return run_dirs
