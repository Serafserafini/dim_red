"""
Pipeline sub-package: unified fetch -> SOAP -> VAE orchestration, driven by
YAML configs, with a shared dataset cache and single-run/sweep entrypoints.
"""

from importlib import import_module

__all__ = [
    "RunConfig",
    "SweepConfig",
    "load_run_config",
    "load_sweep_config",
    "run_single",
    "run_sweep",
]


def __getattr__(name: str):
    """Lazily resolve public pipeline symbols to keep import cost minimal."""
    if name in {"RunConfig", "SweepConfig", "load_run_config", "load_sweep_config"}:
        config = import_module("dim_red.pipeline.config")
        return getattr(config, name)
    if name == "run_single":
        return import_module("dim_red.pipeline.single_run").run_single
    if name == "run_sweep":
        return import_module("dim_red.pipeline.sweep").run_sweep
    raise AttributeError(f"module 'dim_red.pipeline' has no attribute '{name}'")
