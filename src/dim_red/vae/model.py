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


def apply_family_mask(
    spacegroup_logits: Array,
    family_weights: Array,
    family_spacegroup_mask: Array,
    eps: float = 1e-8,
) -> Array:
    """Condition spacegroup logits on a family weighting via additive log-space masking.

    Per-sample plausibility of each spacegroup is ``family_weights @
    family_spacegroup_mask`` (a convex combination of the mask rows for
    hard, one-hot ``family_weights``, or a soft blend for softmax
    ``family_weights``). That plausibility is added in log-space to the raw
    spacegroup logits, driving implausible spacegroups towards ``-inf``
    without needing to concatenate family information into the network input.

    Args:
        spacegroup_logits: Raw (unmasked) spacegroup logits, shape
            ``(batch, n_spacegroup_classes)``.
        family_weights: Either a hard one-hot family encoding (training,
            using the true family label) or softmax family probabilities
            (inference, when the true family is unknown), shape
            ``(batch, n_family_classes)``.
        family_spacegroup_mask: ``1.0`` where a spacegroup was observed
            under a family, ``0.0`` otherwise, shape ``(n_family_classes,
            n_spacegroup_classes)``.
        eps: Additive floor to keep ``log`` finite for fully-masked entries.

    Returns:
        Masked spacegroup logits, same shape as ``spacegroup_logits``.
    """
    plausibility = family_weights @ family_spacegroup_mask
    return spacegroup_logits + jnp.log(plausibility + eps)


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


class ClassifierHead(nn.Module):
    """Small MLP classifying latent codes into discrete classes.

    Deliberately narrow (a single hidden layer) so that its gradient does
    not dominate the encoder relative to the reconstruction + KL terms.
    """

    hidden_dim: int
    n_classes: int

    @nn.compact
    def __call__(self, z: Array) -> Array:
        """Return class logits for a batch of latent codes ``z``."""
        h = nn.relu(nn.Dense(self.hidden_dim)(z))
        return nn.Dense(self.n_classes)(h)


class VAEModule(nn.Module):
    """Core Flax VAE module exposing explicit encode/decode/forward paths.

    Optionally also exposes small auxiliary classifier heads (family and/or
    spacegroup) operating on the sampled latent ``z``, controlled by
    ``n_family_classes``/``n_spacegroup_classes``. Leaving both ``None``
    (the default) reproduces the plain VAE with no auxiliary heads.
    """

    input_dim: int
    encoder_hidden_dim: Sequence[int]
    decoder_hidden_dim: Sequence[int]

    latent_dim: int
    n_family_classes: Optional[int] = None
    n_spacegroup_classes: Optional[int] = None
    head_hidden_dim: int = 16

    def setup(self):
        """Initialize encoder/decoder submodules and any configured auxiliary heads."""
        self.encoder_net = EncoderModule(
            hidden_dim=self.encoder_hidden_dim, latent_dim=self.latent_dim
        )
        self.decoder_net = DecoderModule(
            hidden_dim=self.decoder_hidden_dim, output_dim=self.input_dim
        )
        self.family_head = (
            ClassifierHead(
                hidden_dim=self.head_hidden_dim, n_classes=self.n_family_classes
            )
            if self.n_family_classes is not None
            else None
        )
        self.spacegroup_head = (
            ClassifierHead(
                hidden_dim=self.head_hidden_dim, n_classes=self.n_spacegroup_classes
            )
            if self.n_spacegroup_classes is not None
            else None
        )

    def encode(self, x: Array) -> Tuple[Array, Array]:
        """Run encoder network and return latent mean and log-variance."""
        return self.encoder_net(x)

    def decode(self, z: Array) -> Array:
        """Run decoder network and return reconstruction in input space."""
        return self.decoder_net(z)

    def classify_family(self, z: Array) -> Array:
        """Return raw (unmasked) family logits for latent batch ``z``."""
        if self.family_head is None:
            raise ValueError("Family head is not configured (n_family_classes is None)")
        return self.family_head(z)

    def classify_spacegroup(self, z: Array) -> Array:
        """Return raw (unmasked) spacegroup logits for latent batch ``z``.

        Callers are expected to condition these on family via
        :func:`apply_family_mask` -- this method deliberately returns the
        unmasked logits since the masking weighting (hard true-family at
        train time, soft predicted-family at inference) is a training/
        inference policy choice, not a model architecture detail.
        """
        if self.spacegroup_head is None:
            raise ValueError(
                "Spacegroup head is not configured (n_spacegroup_classes is None)"
            )
        return self.spacegroup_head(z)

    def __call__(self, x: Array, sample_key: Array) -> Tuple[Array, Array, Array]:
        """Full VAE forward pass returning reconstruction and latent stats."""
        mu, logvar = self.encode(x)
        z = reparameterize(sample_key, mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def init_all(self, x: Array, sample_key: Array) -> Tuple[Array, Array, Array]:
        """Trace every submodule, including any configured auxiliary heads.

        Used only as the ``method`` passed to ``Module.init`` so that
        parameters get allocated for the heads too -- ``__call__`` itself
        never touches them, since it must keep returning exactly
        ``(x_recon, mu, logvar)`` regardless of which heads are configured.
        """
        mu, logvar = self.encode(x)
        z = reparameterize(sample_key, mu, logvar)
        x_recon = self.decode(z)
        if self.family_head is not None:
            self.classify_family(z)
        if self.spacegroup_head is not None:
            self.classify_spacegroup(z)
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
        n_family_classes: Optional[int] = None,
        n_spacegroup_classes: Optional[int] = None,
        head_hidden_dim: int = 16,
    ):
        """Initialize module architecture and random model parameters.

        Args:
            input_dim: Number of input features.
            encoder_hidden_dim: Widths of hidden layers in encoder.
            decoder_hidden_dim: Widths of hidden layers in decoder.
            latent_dim: Size of latent representation.
            mirror: Whether to mirror encoder hidden dims in decoder (default: True).
            seed: Random seed used for Flax parameter initialization.
            n_family_classes: If set, adds a small family-classification head
                on ``z``. ``None`` (default) means no auxiliary heads at all.
            n_spacegroup_classes: If set, adds a small spacegroup-
                classification head on ``z``, conditioned on family via
                :func:`apply_family_mask` (applied by the caller, not this
                class). Requires ``n_family_classes`` to also be set.
            head_hidden_dim: Hidden width of each auxiliary head's single
                hidden layer.

        Raises:
            ValueError: If any dimensional argument is non-positive, or if
                ``n_spacegroup_classes`` is set without ``n_family_classes``.
        """
        if n_spacegroup_classes is not None and n_family_classes is None:
            raise ValueError(
                "n_family_classes must be set when n_spacegroup_classes is set "
                "(the spacegroup head is conditioned on family)"
            )
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
        self.n_family_classes = n_family_classes
        self.n_spacegroup_classes = n_spacegroup_classes
        self.head_hidden_dim = head_hidden_dim

        self.module = VAEModule(
            input_dim=input_dim,
            encoder_hidden_dim=self.encoder_hidden_dim,
            decoder_hidden_dim=self.decoder_hidden_dim,
            latent_dim=latent_dim,
            n_family_classes=n_family_classes,
            n_spacegroup_classes=n_spacegroup_classes,
            head_hidden_dim=head_hidden_dim,
        )
        init_x = jnp.zeros((1, input_dim), dtype=jnp.float32)
        rng = jax.random.PRNGKey(seed)
        k1, k2 = jax.random.split(rng)
        # `init_all` (rather than `__call__`) traces the auxiliary heads too,
        # if configured, so their params get allocated here regardless of
        # which heads are present -- see `VAEModule.init_all`.
        variables = self.module.init(
            {"params": k1}, init_x, k2, method=self.module.init_all
        )
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

    def classify_family_with_params(self, params: Params, z: Array) -> Array:
        """Return raw (unmasked) family logits for latent batch ``z``.

        Raises:
            ValueError: If the model was built without a family head.
        """
        return self.module.apply(
            {"params": params},
            jnp.asarray(z, dtype=jnp.float32),
            method=self.module.classify_family,
        )

    def classify_spacegroup_with_params(self, params: Params, z: Array) -> Array:
        """Return raw (unmasked) spacegroup logits for latent batch ``z``.

        Callers combine this with a family weighting via
        :func:`apply_family_mask` before taking a softmax/cross-entropy.

        Raises:
            ValueError: If the model was built without a spacegroup head.
        """
        return self.module.apply(
            {"params": params},
            jnp.asarray(z, dtype=jnp.float32),
            method=self.module.classify_spacegroup,
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

    def classify_family(self, z: Array) -> Array:
        """Return raw (unmasked) family logits using the stored parameters."""
        return self.classify_family_with_params(self.params, z)

    def classify_spacegroup(self, z: Array) -> Array:
        """Return raw (unmasked) spacegroup logits using the stored parameters."""
        return self.classify_spacegroup_with_params(self.params, z)

    def forward(self, x: Array, key: Array) -> Tuple[Array, Array, Array]:
        """Run full VAE forward pass using the internally stored parameters."""
        return self.forward_with_params(self.params, x, key)
