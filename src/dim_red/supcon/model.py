"""
Encoder-only model definitions (Flax Linen) for the Supervised Contrastive
(SupCon) latent-space variant.

No decoder, no KL term, no classifier heads: this package's only job is to
map input feature vectors to a latent ``z`` that a Supervised Contrastive
loss (see ``dim_red.supcon.training``) pulls together for same-label
structures and pushes apart otherwise. Encoding is deterministic (a single
vector, no PRNG key needed at encode time), the same shape as
``dim_red.autoencoder.model.Autoencoder.encode`` -- ``EncoderModule`` below
is a deliberate duplicate of that module's encoder half (not an import) so
this package has no dependency on ``dim_red.vae``/``dim_red.autoencoder``.
"""

from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

Array = jax.Array
Params = dict


class EncoderModule(nn.Module):
    """Flax Linen encoder mapping input vectors to a single latent vector."""

    hidden_dim: Sequence[int]
    latent_dim: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """Encode input features into a deterministic latent vector ``z``."""
        h = x
        for dim in self.hidden_dim:
            h = nn.relu(nn.Dense(dim)(h))
        return nn.Dense(self.latent_dim)(h)


class Encoder:
    """Thin adapter exposing only the encoder interface of a trained ``SupConEncoder``."""

    def __init__(self, supcon_encoder: "SupConEncoder"):
        self._supcon_encoder = supcon_encoder

    def __call__(self, x: Array) -> Array:
        """Return the latent vector ``z`` for input batch ``x``."""
        return self._supcon_encoder.encode(x)


class SupConEncoder:
    """High-level wrapper managing Flax module creation and parameters for a
    pure encoder trained with a Supervised Contrastive loss.

    Keeps a mutable ``params`` field with the latest trained parameters,
    while exposing an ``encode``/``encode_with_params`` pair for inference
    or training loops. No ``decode``, no classifier heads, no PRNG key
    needed past initialization -- see the module docstring.
    """

    def __init__(
        self,
        input_dim: int,
        encoder_hidden_dim: Sequence[int],
        latent_dim: int,
        seed: int = 42,
    ):
        """Initialize module architecture and random model parameters.

        Args:
            input_dim: Number of input features.
            encoder_hidden_dim: Widths of hidden layers in the encoder.
            latent_dim: Size of the latent representation.
            seed: Random seed used for Flax parameter initialization.

        Raises:
            ValueError: If any dimensional argument is non-positive.
        """
        if not isinstance(encoder_hidden_dim, Sequence):
            encoder_hidden_dim = [encoder_hidden_dim]
        if (
            input_dim <= 0
            or any(dim <= 0 for dim in encoder_hidden_dim)
            or latent_dim <= 0
        ):
            raise ValueError("All dimensional arguments must be positive integers")

        self.input_dim = input_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.latent_dim = latent_dim

        self.module = EncoderModule(
            hidden_dim=self.encoder_hidden_dim, latent_dim=latent_dim
        )
        init_x = jnp.zeros((1, input_dim), dtype=jnp.float32)
        rng = jax.random.PRNGKey(seed)
        variables = self.module.init({"params": rng}, init_x)
        self.params = variables["params"]
        self.encoder = Encoder(self)

    def encode_with_params(self, params: Params, x: Array) -> Array:
        """Encode ``x`` using an explicit parameter tree."""
        return self.module.apply({"params": params}, jnp.asarray(x, dtype=jnp.float32))

    def encode(self, x: Array) -> Array:
        """Encode ``x`` using the internally stored model parameters."""
        return self.encode_with_params(self.params, x)
