"""
dim_red - Dimensionality Reduction Package
"""

from importlib import import_module

from dim_red.pca import PCA
from dim_red.utils import standardize

__version__ = "0.1.0"
__all__ = [
    "PCA",
    "standardize",
    "compute_soap",
    "fetch_structures_by_crystal_system",
    "analysis",
    "vae",
    "autoencoder",
    "supcon",
    "cgcnn",
    "mace",
    "pipeline",
    "augmentation",
    "generate",
]


def __getattr__(name: str):
    """Lazily import optional submodules to avoid hard dependency failures."""
    if name == "compute_soap":
        return import_module("dim_red.soap").compute_soap
    if name == "fetch_structures_by_crystal_system":
        return import_module("dim_red.fetch").fetch_structures_by_crystal_system
    if name == "augmentation":
        return import_module("dim_red.augmentation")
    if name == "generate":
        return import_module("dim_red.generate")
    if name == "analysis":
        return import_module("dim_red.analysis")
    if name == "vae":
        return import_module("dim_red.vae")
    if name == "autoencoder":
        return import_module("dim_red.autoencoder")
    if name == "supcon":
        return import_module("dim_red.supcon")
    if name == "cgcnn":
        return import_module("dim_red.cgcnn")
    if name == "mace":
        return import_module("dim_red.mace")
    if name == "pipeline":
        return import_module("dim_red.pipeline")
    raise AttributeError(f"module 'dim_red' has no attribute '{name}'")
