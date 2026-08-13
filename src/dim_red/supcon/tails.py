"""
Tail modules attachable to a SupCon body (``dim_red.supcon.model.SupConEncoder``),
one at a time -- see ``dim_red.supcon.training`` (phase 1) and
``dim_red.supcon.tail_training`` (phase 2) for how each is actually trained.

- ``ProjectionTail``: attached during the body's own (only) training stage --
  trained jointly with the body via the Supervised Contrastive loss
  (``dim_red.supcon.training.supcon_loss``), computed on *this tail's*
  output rather than the body's raw representation ``r`` (Khosla et al.
  2020's ``z = Proj(Enc(x))``). The paper discards the projection head after
  pretraining; this codebase still saves its trained params for
  reproducibility (see ``dim_red.pipeline.single_run``), even though nothing
  downstream ever reloads them.
- ``ClassificationTail``: attached *after* the body is frozen -- trains only
  its own parameters via cross-entropy on family and/or spacegroup labels
  (``dim_red.supcon.tail_training.train_classification_tail``). Mirrors
  ``dim_red.vae.model``'s ``ClassifierHead``/``apply_family_mask`` (verbatim
  duplicates, not imports -- same no-cross-package-dependency convention as
  ``dim_red.supcon.model.EncoderModule``).
- ``VisualizationTail``: also attached after freezing, mutually exclusive
  with ``ClassificationTail``. Projects ``r`` down to 2 or 3 dimensions,
  trained with the same ``supcon_loss`` as ``ProjectionTail``
  (``dim_red.supcon.tail_training.train_visualization_tail``) so the
  low-dimensional projection still respects family/spacegroup neighborhood
  structure, unlike a purely unsupervised post-hoc projection (e.g. UMAP).

Exactly one tail is ever attached to the body at a time. Every tail here is
an independent, separately-initialized Flax wrapper (its own ``params``
pytree), never nested inside ``SupConEncoder``'s own module/param tree --
freezing the body for phase 2 is then simply "don't include its params in
that stage's optimizer" (see ``dim_red.supcon.tail_training``), no
``stop_gradient``/masked-optimizer machinery needed.
"""

from typing import Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

Array = jax.Array
Params = dict


class _MLPTail(nn.Module):
    """Small MLP: a Dense/relu stack, then an unactivated final
    ``Dense(output_dim)``. Shared by ``ProjectionTail``/``VisualizationTail``
    -- structurally and loss-identical (both trained with ``supcon_loss``),
    differing only in output width and which training stage attaches them.
    """

    hidden_dim: Sequence[int]
    output_dim: int

    @nn.compact
    def __call__(self, r: Array) -> Array:
        h = r
        for dim in self.hidden_dim:
            h = nn.relu(nn.Dense(dim)(h))
        return nn.Dense(self.output_dim)(h)


class ProjectionTail:
    """High-level wrapper managing Flax module creation and parameters for
    the projection tail: maps a SupCon body's representation ``r`` into the
    space the contrastive loss is actually computed in during phase-1
    training. Mirrors ``SupConEncoder``'s shape (mutable ``params``,
    ``project``/``project_with_params``).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Sequence[int],
        projection_dim: int,
        seed: int = 42,
    ):
        """Initialize module architecture and random parameters.

        Args:
            input_dim: Width of the body's representation ``r`` (its
                ``latent_dim``).
            hidden_dim: Widths of hidden layers in the projection MLP.
            projection_dim: Size of the space the SupCon loss is computed
                in (Khosla et al. 2020 default: 128).
            seed: Random seed used for Flax parameter initialization.

        Raises:
            ValueError: If any dimensional argument is non-positive.
        """
        if not isinstance(hidden_dim, Sequence):
            hidden_dim = [hidden_dim]
        if input_dim <= 0 or any(dim <= 0 for dim in hidden_dim) or projection_dim <= 0:
            raise ValueError("All dimensional arguments must be positive integers")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim

        self.module = _MLPTail(hidden_dim=hidden_dim, output_dim=projection_dim)
        init_r = jnp.zeros((1, input_dim), dtype=jnp.float32)
        rng = jax.random.PRNGKey(seed)
        variables = self.module.init({"params": rng}, init_r)
        self.params = variables["params"]

    def project_with_params(self, params: Params, r: Array) -> Array:
        """Project representation ``r`` using an explicit parameter tree."""
        return self.module.apply({"params": params}, jnp.asarray(r, dtype=jnp.float32))

    def project(self, r: Array) -> Array:
        """Project representation ``r`` using the internally stored parameters."""
        return self.project_with_params(self.params, r)


class VisualizationTail:
    """High-level wrapper managing Flax module creation and parameters for
    the visualization tail: maps a frozen SupCon body's representation ``r``
    down to 2 or 3 dimensions for plotting, trained with the same
    ``supcon_loss`` as ``ProjectionTail`` (see
    ``dim_red.supcon.tail_training.train_visualization_tail``).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Sequence[int],
        output_dim: int,
        seed: int = 42,
    ):
        """Initialize module architecture and random parameters.

        Args:
            input_dim: Width of the body's representation ``r``.
            hidden_dim: Widths of hidden layers in the visualization MLP.
            output_dim: The plotted dimensionality -- 2 or 3.
            seed: Random seed used for Flax parameter initialization.

        Raises:
            ValueError: If any dimensional argument is non-positive, or
                ``output_dim`` is not 2 or 3.
        """
        if not isinstance(hidden_dim, Sequence):
            hidden_dim = [hidden_dim]
        if input_dim <= 0 or any(dim <= 0 for dim in hidden_dim):
            raise ValueError("All dimensional arguments must be positive integers")
        if output_dim not in (2, 3):
            raise ValueError(f"output_dim must be 2 or 3, got {output_dim!r}")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.module = _MLPTail(hidden_dim=hidden_dim, output_dim=output_dim)
        init_r = jnp.zeros((1, input_dim), dtype=jnp.float32)
        rng = jax.random.PRNGKey(seed)
        variables = self.module.init({"params": rng}, init_r)
        self.params = variables["params"]

    def project_with_params(self, params: Params, r: Array) -> Array:
        """Project representation ``r`` using an explicit parameter tree."""
        return self.module.apply({"params": params}, jnp.asarray(r, dtype=jnp.float32))

    def project(self, r: Array) -> Array:
        """Project representation ``r`` using the internally stored parameters."""
        return self.project_with_params(self.params, r)


def apply_family_mask(
    spacegroup_logits: Array,
    family_weights: Array,
    family_spacegroup_mask: Array,
    eps: float = 1e-8,
) -> Array:
    """Condition spacegroup logits on a family weighting via additive log-space masking.

    Verbatim duplicate of ``dim_red.vae.model.apply_family_mask`` -- kept
    independent so this package has no dependency on ``dim_red.vae``/
    ``dim_red.autoencoder`` (same convention as
    ``dim_red.supcon.model.EncoderModule``).

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


class ClassifierHead(nn.Module):
    """Small MLP classifying representations into discrete classes.

    Verbatim duplicate of ``dim_red.vae.model.ClassifierHead``.
    """

    hidden_dim: int
    n_classes: int

    @nn.compact
    def __call__(self, r: Array) -> Array:
        """Return class logits for a batch of representations ``r``."""
        h = nn.relu(nn.Dense(self.hidden_dim)(r))
        return nn.Dense(self.n_classes)(h)


class ClassificationTailModule(nn.Module):
    """Family and/or spacegroup classifier heads operating on a frozen
    SupCon body's representation ``r``. Mirrors
    ``dim_red.vae.model.VAEModule``'s head setup, minus the encoder/decoder
    it also owns -- this module has no forward pass of its own besides the
    heads.
    """

    hidden_dim: int
    n_family_classes: Optional[int] = None
    n_spacegroup_classes: Optional[int] = None

    def setup(self):
        """Initialize whichever heads are configured."""
        self.family_head = (
            ClassifierHead(hidden_dim=self.hidden_dim, n_classes=self.n_family_classes)
            if self.n_family_classes is not None
            else None
        )
        self.spacegroup_head = (
            ClassifierHead(
                hidden_dim=self.hidden_dim, n_classes=self.n_spacegroup_classes
            )
            if self.n_spacegroup_classes is not None
            else None
        )

    def classify_family(self, r: Array) -> Array:
        """Return raw (unmasked) family logits for representation batch ``r``."""
        if self.family_head is None:
            raise ValueError("Family head is not configured (n_family_classes is None)")
        return self.family_head(r)

    def classify_spacegroup(self, r: Array) -> Array:
        """Return raw (unmasked) spacegroup logits for representation batch ``r``.

        Callers are expected to condition these on family via
        :func:`apply_family_mask` -- see
        ``dim_red.vae.model.VAEModule.classify_spacegroup``.
        """
        if self.spacegroup_head is None:
            raise ValueError(
                "Spacegroup head is not configured (n_spacegroup_classes is None)"
            )
        return self.spacegroup_head(r)

    def init_all(self, r: Array) -> None:
        """Trace every configured head so ``.init()`` allocates their
        params -- this module has no other forward pass to piggyback on
        (unlike ``VAEModule.init_all``, which also traces encode/decode).
        """
        if self.family_head is not None:
            self.classify_family(r)
        if self.spacegroup_head is not None:
            self.classify_spacegroup(r)


class ClassificationTail:
    """High-level wrapper managing Flax module creation and parameters for a
    classification tail attached to a frozen SupCon body's representation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_family_classes: Optional[int] = None,
        n_spacegroup_classes: Optional[int] = None,
        seed: int = 42,
    ):
        """Initialize module architecture and random parameters.

        Args:
            input_dim: Width of the body's (frozen) representation ``r``.
            hidden_dim: Hidden width of each head's single hidden layer.
            n_family_classes: If set, adds a family-classification head.
            n_spacegroup_classes: If set, adds a spacegroup-classification
                head, conditioned on family via :func:`apply_family_mask`
                (applied by the caller). Requires ``n_family_classes`` to
                also be set.
            seed: Random seed used for Flax parameter initialization.

        Raises:
            ValueError: If any dimensional argument is non-positive, if
                neither class count is set, or if ``n_spacegroup_classes``
                is set without ``n_family_classes``.
        """
        if n_family_classes is None and n_spacegroup_classes is None:
            raise ValueError(
                "At least one of n_family_classes/n_spacegroup_classes must be set"
            )
        if n_spacegroup_classes is not None and n_family_classes is None:
            raise ValueError(
                "n_family_classes must be set when n_spacegroup_classes is set "
                "(the spacegroup head is conditioned on family)"
            )
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("All dimensional arguments must be positive integers")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_family_classes = n_family_classes
        self.n_spacegroup_classes = n_spacegroup_classes

        self.module = ClassificationTailModule(
            hidden_dim=hidden_dim,
            n_family_classes=n_family_classes,
            n_spacegroup_classes=n_spacegroup_classes,
        )
        init_r = jnp.zeros((1, input_dim), dtype=jnp.float32)
        rng = jax.random.PRNGKey(seed)
        variables = self.module.init(
            {"params": rng}, init_r, method=self.module.init_all
        )
        self.params = variables["params"]

    def classify_family_with_params(self, params: Params, r: Array) -> Array:
        """Return raw (unmasked) family logits for representation batch ``r``."""
        return self.module.apply(
            {"params": params},
            jnp.asarray(r, dtype=jnp.float32),
            method=self.module.classify_family,
        )

    def classify_spacegroup_with_params(self, params: Params, r: Array) -> Array:
        """Return raw (unmasked) spacegroup logits for representation batch ``r``."""
        return self.module.apply(
            {"params": params},
            jnp.asarray(r, dtype=jnp.float32),
            method=self.module.classify_spacegroup,
        )

    def classify_family(self, r: Array) -> Array:
        """Return raw (unmasked) family logits using the stored parameters."""
        return self.classify_family_with_params(self.params, r)

    def classify_spacegroup(self, r: Array) -> Array:
        """Return raw (unmasked) spacegroup logits using the stored parameters."""
        return self.classify_spacegroup_with_params(self.params, r)
