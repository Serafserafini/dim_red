"""
VAE sub-package for dataset handling, Flax model definition, training and
independent encoder/decoder usage.
"""

from importlib import import_module

__all__ = [
    "VAEDatabase",
    "Encoder",
    "Decoder",
    "VAE",
    "TrainConfig",
    "train_vae",
    "vae_loss",
    "VAEEncoder",
    "VAEDecoder",
    "split_encoder_decoder",
]


def __getattr__(name: str):
    """Lazily resolve public VAE symbols to keep import cost minimal."""
    if name == "VAEDatabase":
        return import_module("dim_red.vae.database").VAEDatabase
    if name in {"Encoder", "Decoder", "VAE"}:
        model = import_module("dim_red.vae.model")
        return getattr(model, name)
    if name in {"TrainConfig", "train_vae", "vae_loss"}:
        training = import_module("dim_red.vae.training")
        return getattr(training, name)
    if name in {"VAEEncoder", "VAEDecoder", "split_encoder_decoder"}:
        codec = import_module("dim_red.vae.codec")
        return getattr(codec, name)
    raise AttributeError(f"module 'dim_red.vae' has no attribute '{name}'")
