"""
Helpers to use a trained Autoencoder as separate encoder and decoder modules.
"""

import jax

from dim_red.autoencoder.model import Autoencoder

Array = jax.Array


class AutoencoderEncoder:
    """Callable wrapper exposing only the encoder path of a trained autoencoder.

    This is useful when downstream code needs a pure encoder interface
    without direct access to the full Autoencoder object.
    """

    def __init__(self, autoencoder: Autoencoder):
        self._autoencoder = autoencoder

    def __call__(self, x: Array) -> Array:
        """Encode batch ``x`` and return the latent vector ``z``."""
        return self._autoencoder.encode(x)


class AutoencoderDecoder:
    """Callable wrapper exposing only the decoder path of a trained autoencoder."""

    def __init__(self, autoencoder: Autoencoder):
        self._autoencoder = autoencoder

    def __call__(self, z: Array) -> Array:
        """Decode latent batch ``z`` into reconstructed input vectors."""
        return self._autoencoder.decode(z)


def split_encoder_decoder(
    autoencoder: Autoencoder,
) -> tuple[AutoencoderEncoder, AutoencoderDecoder]:
    """Split a trained Autoencoder into independent encoder/decoder callables.

    Args:
        autoencoder: Trained (or initialized) Autoencoder instance.

    Returns:
        Tuple ``(encoder, decoder)`` where each element is a lightweight
        callable wrapper around the corresponding Autoencoder path.
    """
    return AutoencoderEncoder(autoencoder), AutoencoderDecoder(autoencoder)
