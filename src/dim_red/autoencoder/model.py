"""
Classic (non-variational) Autoencoder model definitions using Flax Linen.

Deliberately mirrors ``dim_red.vae.model``'s structure and public API
(``Autoencoder.encode``/``decode``/``forward``, ``*_with_params`` variants,
optional family/spacegroup auxiliary heads on the latent code) so the two
are interchangeable from the pipeline's point of view -- the only real
difference is that encoding here is a single deterministic vector, not a
sampled ``(mu, logvar)`` posterior, so there is no KL term and no PRNG key
needed at encode/decode/forward time (only at initialization).
"""

from typing import Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

Array = jax.Array
Params = dict


def apply_family_mask(
    spacegroup_logits: Array,
    family_weights: Array,
    family_spacegroup_mask: Array,
    eps: float = 1e-8,
) -> Array:
    """Condition spacegroup logits on a family weighting via additive log-space masking.

    Identical computation to ``dim_red.vae.model.apply_family_mask`` --
    duplicated here (rather than imported) so this package has no dependency
    on ``dim_red.vae``. See that module for the full explanation.
    """
    plausibility = family_weights @ family_spacegroup_mask
    return spacegroup_logits + jnp.log(plausibility + eps)


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
    not dominate the encoder relative to the reconstruction term.
    """

    hidden_dim: int
    n_classes: int

    @nn.compact
    def __call__(self, z: Array) -> Array:
        """Return class logits for a batch of latent codes ``z``."""
        h = nn.relu(nn.Dense(self.hidden_dim)(z))
        return nn.Dense(self.n_classes)(h)


class AutoencoderModule(nn.Module):
    """Core Flax autoencoder module exposing explicit encode/decode/forward paths.

    Optionally also exposes small auxiliary classifier heads (family and/or
    spacegroup) operating on the latent ``z``, controlled by
    ``n_family_classes``/``n_spacegroup_classes``. Leaving both ``None``
    (the default) reproduces the plain autoencoder with no auxiliary heads.
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

    def encode(self, x: Array) -> Array:
        """Run encoder network and return the deterministic latent vector."""
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

    def __call__(self, x: Array) -> Tuple[Array, Array]:
        """Full autoencoder forward pass returning reconstruction and latent code."""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

    def init_all(self, x: Array) -> Tuple[Array, Array]:
        """Trace every submodule, including any configured auxiliary heads.

        Used only as the ``method`` passed to ``Module.init`` so that
        parameters get allocated for the heads too -- ``__call__`` itself
        never touches them, since it must keep returning exactly
        ``(x_recon, z)`` regardless of which heads are configured.
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        if self.family_head is not None:
            self.classify_family(z)
        if self.spacegroup_head is not None:
            self.classify_spacegroup(z)
        return x_recon, z


class Encoder:
    """Thin adapter exposing only the encoder interface of a trained autoencoder."""

    def __init__(self, autoencoder: "Autoencoder"):
        self._autoencoder = autoencoder

    def __call__(self, x: Array) -> Array:
        """Return the latent vector ``z`` for input batch ``x``."""
        return self._autoencoder.encode(x)


class Decoder:
    """Thin adapter exposing only the decoder interface of a trained autoencoder."""

    def __init__(self, autoencoder: "Autoencoder"):
        self._autoencoder = autoencoder

    def __call__(self, z: Array) -> Array:
        """Return reconstructed inputs from latent batch ``z``."""
        return self._autoencoder.decode(z)


class Autoencoder:
    """High-level autoencoder wrapper managing Flax module creation and parameters.

    This class keeps a mutable ``params`` field with the latest trained
    parameters, while exposing convenient ``encode``, ``decode`` and ``forward``
    methods for inference or training loops. Unlike ``dim_red.vae.model.VAE``,
    encoding is deterministic: no PRNG key is needed at encode/decode/forward
    time, only at construction (for parameter initialization).
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

        self.module = AutoencoderModule(
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
        # `init_all` (rather than `__call__`) traces the auxiliary heads too,
        # if configured, so their params get allocated here regardless of
        # which heads are present -- see `AutoencoderModule.init_all`.
        variables = self.module.init(
            {"params": rng}, init_x, method=self.module.init_all
        )
        self.params = variables["params"]
        self.encoder = Encoder(self)
        self.decoder = Decoder(self)

    def encode_with_params(self, params: Params, x: Array) -> Array:
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

    def forward_with_params(self, params: Params, x: Array) -> Tuple[Array, Array]:
        """Run a full autoencoder pass with explicit parameters."""
        x_arr = jnp.asarray(x, dtype=jnp.float32)
        return self.module.apply({"params": params}, x_arr)

    def encode(self, x: Array) -> Array:
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

    def forward(self, x: Array) -> Tuple[Array, Array]:
        """Run full autoencoder forward pass using the internally stored parameters."""
        return self.forward_with_params(self.params, x)
