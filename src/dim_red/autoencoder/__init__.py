"""
Autoencoder sub-package: a deterministic (non-variational) counterpart to
``dim_red.vae``, sharing the same dataset container
(``dim_red.vae.database.VAEDatabase``, re-exported here too) since dataset
handling has nothing model-specific about it. Mirrors ``dim_red.vae``'s
public API shape (model, training, codec) so the pipeline can pick either
one interchangeably via config.
"""

from importlib import import_module

__all__ = [
    "VAEDatabase",
    "Encoder",
    "Decoder",
    "Autoencoder",
    "TrainConfig",
    "train_autoencoder",
    "autoencoder_loss",
    "AutoencoderEncoder",
    "AutoencoderDecoder",
    "split_encoder_decoder",
]


def __getattr__(name: str):
    """Lazily resolve public autoencoder symbols to keep import cost minimal."""
    if name == "VAEDatabase":
        return import_module("dim_red.vae.database").VAEDatabase
    if name in {"Encoder", "Decoder", "Autoencoder"}:
        model = import_module("dim_red.autoencoder.model")
        return getattr(model, name)
    if name in {"TrainConfig", "train_autoencoder", "autoencoder_loss"}:
        training = import_module("dim_red.autoencoder.training")
        return getattr(training, name)
    if name in {"AutoencoderEncoder", "AutoencoderDecoder", "split_encoder_decoder"}:
        codec = import_module("dim_red.autoencoder.codec")
        return getattr(codec, name)
    raise AttributeError(f"module 'dim_red.autoencoder' has no attribute '{name}'")
