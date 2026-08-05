"""
SupCon sub-package: an encoder-only counterpart to ``dim_red.vae``/
``dim_red.autoencoder``, sharing the same dataset container
(``dim_red.vae.database.VAEDatabase``, re-exported here too). No decoder, no
KL term, no classifier heads -- the encoder is trained directly with a
Supervised Contrastive loss on family/spacegroup labels so that structures
with similar labels end up nearby in latent space. See ``dim_red.supcon.model``
and ``dim_red.supcon.training`` for details.
"""

from importlib import import_module

__all__ = [
    "VAEDatabase",
    "Encoder",
    "SupConEncoder",
    "TrainConfig",
    "train_supcon",
    "supcon_loss",
    "norm_penalty",
    "balanced_batch_indices",
    "iter_balanced_batches",
]


def __getattr__(name: str):
    """Lazily resolve public supcon symbols to keep import cost minimal."""
    if name == "VAEDatabase":
        return import_module("dim_red.vae.database").VAEDatabase
    if name in {"Encoder", "SupConEncoder"}:
        model = import_module("dim_red.supcon.model")
        return getattr(model, name)
    if name in {"TrainConfig", "train_supcon", "supcon_loss", "norm_penalty"}:
        training = import_module("dim_red.supcon.training")
        return getattr(training, name)
    if name in {"balanced_batch_indices", "iter_balanced_batches"}:
        sampling = import_module("dim_red.supcon.sampling")
        return getattr(sampling, name)
    raise AttributeError(f"module 'dim_red.supcon' has no attribute '{name}'")
