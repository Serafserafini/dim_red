"""
Variational Autoencoder (VAE) model definitions using Flax Linen.
"""

from typing import Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

Array = jax.Array
Params = dict


def reparameterize(key: Array, mu: Array, logvar: Array) -> Array:
    """Sample latent codes with the reparameterization trick.

    Args:
        key: JAX PRNG key used to sample noise.
        mu: Mean tensor of the approximate posterior ``q(z|x)``.
        logvar: Log-variance tensor of the approximate posterior ``q(z|x)``.

    Returns:
        A sampled latent tensor ``z`` with the same shape as ``mu``.
    """
    std = jnp.exp(0.5 * logvar)
    eps = jax.random.normal(key, std.shape, dtype=std.dtype)
    return mu + eps * std


class EncoderModule(nn.Module):
    """Flax Linen encoder mapping input vectors to latent distribution params."""

    hidden_dim: Sequence[int]
    latent_dim: int

    @nn.compact
    def __call__(self, x: Array) -> Tuple[Array, Array]:
        """Encode input features into ``(mu, logvar)`` tensors."""
        h = x
        for dim in self.hidden_dim:
            h = nn.relu(nn.Dense(dim)(h))
        mu = nn.Dense(self.latent_dim)(h)
        logvar = nn.Dense(self.latent_dim)(h)
        return mu, logvar


class DecoderModule(nn.Module):
    """Flax Linen decoder mapping latent vectors back to feature space."""

    hidden_dim: Sequence[int]
    output_dim: int

    @nn.compact
    def __call__(self, z: Array) -> Array:
        """Decode latent vectors into reconstructed input features."""
        h = z
        for dim in self.hidden_dim:
            h = nn.relu(nn.Dense(dim)(h))
        return nn.Dense(self.output_dim)(h)


class VAEModule(nn.Module):
    """Core Flax VAE module exposing explicit encode/decode/forward paths."""

    input_dim: int
    encoder_hidden_dim: Sequence[int]
    decoder_hidden_dim: Sequence[int]

    latent_dim: int

    def setup(self):
        """Initialize encoder and decoder submodules."""
        self.encoder_net = EncoderModule(
            hidden_dim=self.encoder_hidden_dim, latent_dim=self.latent_dim
        )
        self.decoder_net = DecoderModule(
            hidden_dim=self.decoder_hidden_dim, output_dim=self.input_dim
        )

    def encode(self, x: Array) -> Tuple[Array, Array]:
        """Run encoder network and return latent mean and log-variance."""
        return self.encoder_net(x)

    def decode(self, z: Array) -> Array:
        """Run decoder network and return reconstruction in input space."""
        return self.decoder_net(z)

    def __call__(self, x: Array, sample_key: Array) -> Tuple[Array, Array, Array]:
        """Full VAE forward pass returning reconstruction and latent stats."""
        mu, logvar = self.encode(x)
        z = reparameterize(sample_key, mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


class Encoder:
    """Thin adapter exposing only the encoder interface of a trained VAE."""

    def __init__(self, vae: "VAE"):
        self._vae = vae

    def __call__(self, x: Array) -> Tuple[Array, Array]:
        """Return ``(mu, logvar)`` for input batch ``x``."""
        return self._vae.encode(x)


class Decoder:
    """Thin adapter exposing only the decoder interface of a trained VAE."""

    def __init__(self, vae: "VAE"):
        self._vae = vae

    def __call__(self, z: Array) -> Array:
        """Return reconstructed inputs from latent batch ``z``."""
        return self._vae.decode(z)


class VAE:
    """High-level VAE wrapper managing Flax module creation and parameters.

    This class keeps a mutable ``params`` field with the latest trained
    parameters, while exposing convenient ``encode``, ``decode`` and ``forward``
    methods for inference or training loops.
    """

    def __init__(
        self,
        input_dim: int,
        encoder_hidden_dim: Sequence[int],
        decoder_hidden_dim: Optional[Sequence[int]],
        latent_dim: int,
        mirror: Optional[bool] = True,
        seed: int = 42,
    ):
        """Initialize module architecture and random model parameters.

        Args:
            input_dim: Number of input features.
            encoder_hidden_dim: Widths of hidden layers in encoder.
            decoder_hidden_dim: Widths of hidden layers in decoder.
            latent_dim: Size of latent representation.
            mirror: Whether to mirror encoder hidden dims in decoder (default: True).
            seed: Random seed used for Flax parameter initialization.

        Raises:
            ValueError: If any dimensional argument is non-positive.
        """
        if not isinstance(encoder_hidden_dim, Sequence):
            encoder_hidden_dim = [encoder_hidden_dim]
        if decoder_hidden_dim is not None and not isinstance(
            decoder_hidden_dim, Sequence
        ):
            decoder_hidden_dim = [decoder_hidden_dim]

        if mirror and decoder_hidden_dim is not None:
            raise ValueError("Cannot specify decoder_hidden_dim when mirror is True")
        if not mirror and decoder_hidden_dim is None:
            raise ValueError("Must specify decoder_hidden_dim when mirror is False")
        if mirror and decoder_hidden_dim is None:
            decoder_hidden_dim = encoder_hidden_dim[::-1]
        if (
            input_dim <= 0
            or any(dim <= 0 for dim in encoder_hidden_dim)
            or any(dim <= 0 for dim in decoder_hidden_dim)
            or latent_dim <= 0
        ):
            raise ValueError("All dimensional arguments must be positive integers")

        self.input_dim = input_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim

        self.latent_dim = latent_dim

        self.module = VAEModule(
            input_dim=input_dim,
            encoder_hidden_dim=self.encoder_hidden_dim,
            decoder_hidden_dim=self.decoder_hidden_dim,
            latent_dim=latent_dim,
        )
        init_x = jnp.zeros((1, input_dim), dtype=jnp.float32)
        rng = jax.random.PRNGKey(seed)
        k1, k2 = jax.random.split(rng)
        variables = self.module.init({"params": k1}, init_x, k2)
        self.params = variables["params"]
        self.encoder = Encoder(self)
        self.decoder = Decoder(self)

    @staticmethod
    def reparameterize(key: Array, mu: Array, logvar: Array) -> Array:
        """Delegate to the module-level reparameterization helper."""
        return reparameterize(key, mu, logvar)

    def encode_with_params(self, params: Params, x: Array) -> Tuple[Array, Array]:
        """Encode ``x`` using an explicit parameter tree."""
        return self.module.apply(
            {"params": params},
            jnp.asarray(x, dtype=jnp.float32),
            method=self.module.encode,
        )

    def decode_with_params(self, params: Params, z: Array) -> Array:
        """Decode ``z`` using an explicit parameter tree."""
        return self.module.apply(
            {"params": params},
            jnp.asarray(z, dtype=jnp.float32),
            method=self.module.decode,
        )

    def forward_with_params(
        self, params: Params, x: Array, key: Array
    ) -> Tuple[Array, Array, Array]:
        """Run a full VAE pass with explicit parameters and RNG key."""
        x_arr = jnp.asarray(x, dtype=jnp.float32)
        return self.module.apply({"params": params}, x_arr, key)

    def encode(self, x: Array) -> Tuple[Array, Array]:
        """Encode ``x`` using the internally stored model parameters."""
        return self.encode_with_params(self.params, x)

    def decode(self, z: Array) -> Array:
        """Decode ``z`` using the internally stored model parameters."""
        return self.decode_with_params(self.params, z)

    def forward(self, x: Array, key: Array) -> Tuple[Array, Array, Array]:
        """Run full VAE forward pass using the internally stored parameters."""
        return self.forward_with_params(self.params, x, key)
