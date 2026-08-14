"""
CGCNN sub-package: a hand-written JAX/Flax reimplementation of Crystal
Graph Convolutional Neural Networks (Xie & Grossman 2018) -- graph
construction (``dim_red.cgcnn.graph``, pure NumPy/ASE, no jax dependency),
the graph-conv model (``dim_red.cgcnn.model``), and a single-phase joint
body+classifier-head training loop via cross-entropy on family/spacegroup
labels (``dim_red.cgcnn.training``) -- mirrors ``dim_red.autoencoder``'s
aux-heads training pattern, not ``dim_red.supcon``'s two-phase body/tail
pattern. Has its own ``GraphDatabase`` (not ``dim_red.vae.database.VAEDatabase``,
since a graph batch is 5 parallel padded arrays, not one flat matrix) and no
dependency on ``vae``/``autoencoder``/``supcon`` -- ``ClassifierHead``/
``apply_family_mask`` are duplicated here, not imported.
"""

from importlib import import_module

__all__ = [
    "GraphDatabase",
    "atoms_to_graph",
    "atoms_list_to_graph_arrays",
    "Encoder",
    "CGCNNEncoder",
    "apply_family_mask",
    "TrainConfig",
    "train_cgcnn",
]


def __getattr__(name: str):
    """Lazily resolve public cgcnn symbols to keep import cost minimal."""
    if name == "GraphDatabase":
        return import_module("dim_red.cgcnn.database").GraphDatabase
    if name in {"atoms_to_graph", "atoms_list_to_graph_arrays"}:
        graph = import_module("dim_red.cgcnn.graph")
        return getattr(graph, name)
    if name in {"Encoder", "CGCNNEncoder", "apply_family_mask"}:
        model = import_module("dim_red.cgcnn.model")
        return getattr(model, name)
    if name in {"TrainConfig", "train_cgcnn"}:
        training = import_module("dim_red.cgcnn.training")
        return getattr(training, name)
    raise AttributeError(f"module 'dim_red.cgcnn' has no attribute '{name}'")
