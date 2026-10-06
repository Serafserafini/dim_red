"""
SupCon sub-package: an encoder-only model trained with a Supervised
Contrastive loss, sharing the dataset container ``dim_red.dataset.FeatureDatabase``
(re-exported here too). No decoder -- just a body (``SupConEncoder``) and, at
any given moment, exactly one attachable tail (``dim_red.supcon.tails``): a
``ProjectionTail`` trained jointly with the body via a Supervised Contrastive
loss (phase 1, see ``dim_red.supcon.training``), or -- after the body is
frozen -- a ``ClassificationTail``/``VisualizationTail`` trained on top of it
(phase 2, see ``dim_red.supcon.tail_training``).
"""

from importlib import import_module

__all__ = [
    "FeatureDatabase",
    "SupConEncoder",
    "ProjectionTail",
    "ClassificationTail",
    "VisualizationTail",
    "TrainConfig",
    "training_first_phase",
    "supcon_loss",
    "norm_penalty",
    "balanced_batch_indices",
    "iter_balanced_batches",
    "TailTrainConfig",
    "train_classification_tail",
    "train_visualization_tail",
    "Batching",
    "StackConfig",
    "SingleStack",
]


def __getattr__(name: str):
    """Lazily resolve public supcon symbols to keep import cost minimal."""
    if name == "FeatureDatabase":
        return import_module("dim_red.dataset").FeatureDatabase
    if name == "SupConEncoder":
        return import_module("dim_red.supcon.model").SupConEncoder
    if name in {
        "ProjectionTail",
        "ClassificationTail",
        "VisualizationTail",
    }:
        tails = import_module("dim_red.supcon.tails")
        return getattr(tails, name)
    if name in {"TrainConfig", "training_first_phase", "supcon_loss", "norm_penalty"}:
        training = import_module("dim_red.supcon.training")
        return getattr(training, name)
    if name in {"balanced_batch_indices", "iter_balanced_batches"}:
        sampling = import_module("dim_red.supcon.sampling")
        return getattr(sampling, name)
    if name in {
        "TailTrainConfig",
        "train_classification_tail",
        "train_visualization_tail",
    }:
        tail_training = import_module("dim_red.supcon.tail_training")
        return getattr(tail_training, name)
    if name in {"Batching", "StackConfig", "SingleStack"}:
        stack = import_module("dim_red.supcon.stack")
        return getattr(stack, name)
    raise AttributeError(f"module 'dim_red.supcon' has no attribute '{name}'")
