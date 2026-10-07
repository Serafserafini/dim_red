"""Grid sweeps over ``FullStack`` configs: the Cartesian product of however
many dotted-path axes ``SweepConfig.grid`` declares (e.g.
``family.encoder.latent_dim``, ``experts.defaults.train.epochs``), one
``FullStack`` run (body + heads ``default``) per combination under a fresh
``<output_dir>/<date>-<n>/`` directory."""

from __future__ import annotations

import copy
import dataclasses
import itertools
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import yaml

from dim_red.pipeline.config import SweepConfig, _set_dotted
from dim_red.pipeline.full_stack import DEFAULT_HEADS_NAME, FullStack
from dim_red.pipeline.full_stack_config import full_stack_config_from_dict

logger = logging.getLogger("dim_red.pipeline")

_SWEEP_DIR_RE = re.compile(r"^\d{8}-(\d+)$")


def _next_sweep_dir(output_dir: Path) -> Path:
    """``<output_dir>/<date>-<n>`` where ``n`` is one more than the highest
    sibling number already present (so concurrent sweeps never collide)."""
    today = datetime.now().strftime("%Y%m%d")
    last_n = 0
    if output_dir.exists():
        for p in output_dir.iterdir():
            m = _SWEEP_DIR_RE.match(p.name) if p.is_dir() else None
            if m:
                last_n = max(last_n, int(m.group(1)))
    sweep_dir = output_dir / f"{today}-{last_n + 1}"
    sweep_dir.mkdir(parents=True, exist_ok=False)
    return sweep_dir


def _slug(value) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "-", str(value)).strip("-")


def _run_name(base_name: str, keys: List[str], combo) -> str:
    parts = [
        f"{key.rsplit('.', 1)[-1]}-{_slug(value)}" for key, value in zip(keys, combo)
    ]
    return "_".join([base_name, *parts])


def run_sweep(
    sweep: SweepConfig, cache_dir: Optional[Union[str, Path]] = None
) -> List[Path]:
    """Expand ``sweep.grid`` over ``sweep.base`` and run every combination
    sequentially. Every combination is parsed (and so validated) before the
    first one trains. Returns the run directories in grid order."""
    if "model_kind" not in sweep.base and "model" in sweep.base:
        raise ValueError(
            "dimred-sweep only supports FullStack configs (model_kind: "
            "supcon|supcon_mace); run cgcnn configs with dimred-run"
        )
    keys = list(sweep.grid)
    value_lists = [sweep.grid[k] for k in keys]
    combos = list(itertools.product(*value_lists)) if keys else [()]
    configs = []
    for combo in combos:
        d = copy.deepcopy(sweep.base)
        for key, value in zip(keys, combo):
            _set_dotted(d, key, value)
        config = full_stack_config_from_dict(d)
        if keys:
            config = dataclasses.replace(
                config, name=_run_name(config.name, keys, combo)
            )
        configs.append(config)

    output_dir = Path(sweep.output_dir)
    resolved_cache_dir = Path(cache_dir) if cache_dir else output_dir / "_dataset_cache"
    sweep_dir = _next_sweep_dir(output_dir)
    with open(sweep_dir / "sweep.yaml", "w") as f:
        yaml.safe_dump({"base": sweep.base, "grid": sweep.grid}, f, sort_keys=False)
    logger.info(
        "Starting sweep: %d axis/axes (%s) = %d run(s), saved under %s",
        len(keys),
        sorted(keys),
        len(configs),
        sweep_dir,
    )

    run_dirs = []
    for i, config in enumerate(configs, start=1):
        logger.info("[%d/%d] %s", i, len(configs), config.name)
        config = dataclasses.replace(config, output_dir=str(sweep_dir))
        full_stack = FullStack.create(config, cache_dir=resolved_cache_dir)
        full_stack.fit_body()
        full_stack.fit_heads(DEFAULT_HEADS_NAME)
        run_dirs.append(full_stack.run_dir)
    logger.info("Sweep complete: %d run(s) saved under %s", len(run_dirs), sweep_dir)
    return run_dirs
