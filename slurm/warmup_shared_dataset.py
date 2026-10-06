"""One-off dataset-cache warm-up for slurm/tune_all_models.sbatch.

slurm/tuning_sweep_supcon.yaml's base config defines the pyxtal + soap +
augmentation block, hashed into the dataset-cache key
(dim_red.pipeline.dataset_cache._pyxtal_cache_key). The ~35000-structure SOAP
computation (pure dscribe/CPU work, no jax/GPU involved) is better done once
in a lightweight CPU job than inside the GPU sweep task.

This script builds+caches that shared dataset exactly once, into a single
directory, from a lightweight prerequisite job (see
warmup_shared_dataset.sbatch) that runs BEFORE the array -- so the supcon
sweep task hits a cache and never touches SOAP itself. cgcnn is
deliberately not included: it featurizes with graphs, not SOAP, and keeps
its own separate cache.

Usage:
    python slurm/warmup_shared_dataset.py <cache_dir> <sweep_config> [<sweep_config> ...]

Any tuning_sweep_*.yaml with the same dataset-defining settings works as a
<sweep_config>; passing more than one is only a safety check -- this script refuses to proceed if
they don't all resolve to the same cache key, rather than silently building
whichever came first and leaving the others to rebuild anyway.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from dim_red.pipeline.config import expand_sweep, load_sweep_config
from dim_red.pipeline.dataset_cache import (
    _pyxtal_cache_key,
    _resolve_augmentation,
    build_dataset_for_run,
)

logger = logging.getLogger("dim_red.pipeline")


def main(argv=None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) < 2:
        print(__doc__)
        sys.exit(1)

    cache_dir = Path(argv[0])
    sweep_paths = argv[1:]

    # sweep.base is a raw dict (grid overrides get applied on top of it per
    # combination) -- expand_sweep(...)[0] gives back a real RunConfig for
    # this sweep's first grid combination. Since the grid axes in these
    # configs never touch pyxtal/soap/augmentation, every combination shares
    # the same dataset-defining settings, so the first one is representative.
    configs = [expand_sweep(load_sweep_config(p))[0] for p in sweep_paths]
    keys = set()
    for path, config in zip(sweep_paths, configs):
        seed = config.pyxtal.seed if config.pyxtal.seed is not None else config.seed
        key = _pyxtal_cache_key(
            config.pyxtal, seed, config.soap.as_kwargs(), _resolve_augmentation(config)
        )
        keys.add(key)
        logger.info("%s -> cache key %s", path, key)

    if len(keys) > 1:
        raise SystemExit(
            f"Given sweep configs don't share a dataset cache key ({sorted(keys)}) -- "
            "their pyxtal/soap/augmentation settings differ, so they can't share a "
            "warm-up build. Give each its own --cache-dir instead."
        )

    build_dataset_for_run(configs[0], cache_dir=cache_dir)
    logger.info("Shared dataset cache ready at %s", cache_dir)


if __name__ == "__main__":
    main()
