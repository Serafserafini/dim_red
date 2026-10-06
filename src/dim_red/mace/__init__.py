"""
MACE sub-package: a frozen, pretrained MACE (equivariant message-passing,
3-body/angular interactions -- Batatia et al. 2022) feature extractor,
wrapping the external ``mace_jax`` package (https://github.com/ACEsuit/mace-jax)
rather than reimplementing it -- the frozen model wrapper
(``dim_red.mace.model``). Unlike ``dim_red.cgcnn``, there is no
``training.py``: the body is never trained by this codebase, only loaded
(pretrained weights, converted from a Torch foundation model checkpoint
outside this package's runtime) and run forward. Classification on top of
its embeddings happens entirely via ``dim_red.pipeline.tail_training``
(phase 2), the same architecture-agnostic tail machinery ``supcon``/
``cgcnn`` already use.
"""

from importlib import import_module

__all__ = [
    "MaceEncoder",
    "load_frozen_checkpoint",
]


def __getattr__(name: str):
    """Lazily resolve public mace symbols to keep import cost minimal."""
    if name in {"MaceEncoder", "load_frozen_checkpoint"}:
        model = import_module("dim_red.mace.model")
        return getattr(model, name)
    raise AttributeError(f"module 'dim_red.mace' has no attribute '{name}'")
