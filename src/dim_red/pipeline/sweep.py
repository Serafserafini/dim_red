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

from dim_red.pipeline.config import RunConfig, SweepConfig, expand_sweep
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


def _dataset_size_desc(config: RunConfig) -> str:
    """Configured dataset-size for a run's data source, without actually
    building it (the exact count is only known after fetch/generation --
    this reports the config that determines it).
    """
    if config.data_source == "fetch":
        fc = config.fetch
        return f"limit_per_system={fc.limit_per_system} x {len(fc.crystal_systems)} systems"
    pc = config.pyxtal
    if pc.structures_per_spacegroup is not None:
        return f"structures_per_spacegroup={pc.structures_per_spacegroup}"
    return f"structures_per_family={pc.structures_per_family}"


def _architecture_desc(config: RunConfig) -> str:
    vae = config.vae
    decoder = vae.decoder_hidden_dim
    if decoder is None:
        decoder = "mirror" if vae.mirror else "same-as-encoder"
    return f"encoder={vae.encoder_hidden_dim} latent={vae.latent_dim} decoder={decoder}"


def _summarize_field(values: List[str]) -> str:
    """A single display value if every run in the sweep agrees, otherwise a
    ``varies: ...`` listing -- happens when the field itself is a swept axis
    (e.g. ``vae.encoder_hidden_dim``), in which case the actual per-value
    breakdown is already visible in the "Sweep axes" line.
    """
    unique = sorted(set(values))
    if len(unique) == 1:
        return unique[0]
    return "varies: " + " | ".join(unique)


def _write_sweep_readme(
    output_dir: Path, sweep_dir: Path, sweep: SweepConfig, run_configs: List[RunConfig]
) -> None:
    """Append a summary entry for this sweep invocation to
    ``<output_dir>/README.md`` -- creating the file (with a header) the first
    time a sweep is sent to this ``output_dir``, and just appending a new
    entry (never overwriting prior ones) on every subsequent sweep.
    """
    readme_path = output_dir / "README.md"
    output_dir.mkdir(parents=True, exist_ok=True)
    is_new = not readme_path.exists()

    entry = (
        f"## {sweep_dir.name}\n\n"
        f"- Model: {_summarize_field([c.model_kind for c in run_configs])}\n"
        f"- Dataset elements: "
        f"{_summarize_field([_dataset_size_desc(c) for c in run_configs])}\n"
        f"- Dataset source: {_summarize_field([c.data_source for c in run_configs])}\n"
        f"- Architecture: "
        f"{_summarize_field([_architecture_desc(c) for c in run_configs])}\n"
        f"- Sweep axes: {', '.join(sorted(sweep.grid)) if sweep.grid else '(none)'}\n\n"
    )
    with open(readme_path, "a") as f:
        if is_new:
            f.write("# Sweeps\n\n")
        f.write(entry)


def run_sweep(
    sweep: SweepConfig, cache_dir: Optional[Union[str, Path]] = None
) -> List[Path]:
    """Expand a SweepConfig's grid and run each combination sequentially.

    Each invocation gets its own ``<output_dir>/<date>-<n>`` directory (see
    ``_next_sweep_dir``); individual run folders inside it are named from
    their swept hyperparameters only (see ``make_run_name``). The dataset
    cache stays shared at ``<output_dir>/_dataset_cache`` across sweep
    invocations, since it doesn't depend on which sweep produced it.

    Also appends a summary entry for this sweep (model kind, dataset size,
    dataset source, architecture, and swept axes) to ``<output_dir>/README.md``,
    creating that file the first time a sweep lands in ``output_dir`` (see
    ``_write_sweep_readme``).

    Returns:
        The list of run directories created, in grid order.
    """
    run_configs = expand_sweep(sweep)
    output_dir = Path(sweep.output_dir)
    resolved_cache_dir = Path(cache_dir) if cache_dir else output_dir / "_dataset_cache"
    sweep_dir = _next_sweep_dir(output_dir)
    _write_sweep_readme(output_dir, sweep_dir, sweep, run_configs)

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
