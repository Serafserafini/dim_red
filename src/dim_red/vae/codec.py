"""
Helpers to use a trained VAE as separate encoder and decoder modules.
"""

from typing import Tuple
import jax
from dim_red.vae.model import VAE


Array = jax.Array


class VAEEncoder:
    """Callable wrapper exposing only the encoder path of a trained VAE.

    This is useful when downstream code needs a pure encoder interface
    without direct access to the full VAE object.
    """

    def __init__(self, vae: VAE):
        self._vae = vae

    def __call__(self, x: Array) -> Tuple[Array, Array]:
        """Encode batch ``x`` and return ``(mu, logvar)``."""
        return self._vae.encode(x)


class VAEDecoder:
    """Callable wrapper exposing only the decoder path of a trained VAE."""

    def __init__(self, vae: VAE):
        self._vae = vae

    def __call__(self, z: Array) -> Array:
        """Decode latent batch ``z`` into reconstructed input vectors."""
        return self._vae.decode(z)


def split_encoder_decoder(vae: VAE) -> tuple[VAEEncoder, VAEDecoder]:
    """Split a trained VAE into independent encoder/decoder callables.

    Args:
        vae: Trained (or initialized) VAE instance.

    Returns:
        Tuple ``(encoder, decoder)`` where each element is a lightweight
        callable wrapper around the corresponding VAE path.
    """
    return VAEEncoder(vae), VAEDecoder(vae)
