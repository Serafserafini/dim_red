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
- ``VisualizationTail``: attached *after* the body is frozen. Projects ``r``
  down to 2 or 3 dimensions, trained with the same ``supcon_loss`` as
  ``ProjectionTail`` (``dim_red.supcon.tail_training.train_visualization_tail``)
  so the low-dimensional projection still respects family/spacegroup
  neighborhood structure, unlike a purely unsupervised post-hoc projection
  (e.g. UMAP). It is the same model as ``ProjectionTail`` -- an MLP on ``r``
  -- the only differences being that it is trained *after* the body is
  frozen rather than jointly with it, and that its output is 2 or 3 wide.
- ``ClassificationTail``: attached *after* the body is frozen -- an MLP
  classifier trained with cross-entropy on whatever labels it is given (family
  labels for the family-level tail, local spacegroup ids for a per-family
  expert; see ``dim_red.supcon.tail_training.train_classification_tail``).

Exactly one tail is ever attached to the body at a time. Every tail here is
an independent, separately-initialized Flax wrapper (its own ``params``
pytree), never nested inside ``SupConEncoder``'s own module/param tree --
freezing the body for phase 2 is then simply "don't include its params in
that stage's optimizer" (see ``dim_red.supcon.tail_training``), no
``stop_gradient``/masked-optimizer machinery needed.
"""

from typing import Sequence, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax import serialization

Array = jax.Array
Params = dict


def _as_dims(hidden_dim: Union[int, Sequence[int]]) -> tuple:
    """``hidden_dim`` as a tuple of layer widths (an int means one layer)."""
    return (hidden_dim,) if isinstance(hidden_dim, int) else tuple(hidden_dim)


class MLP(nn.Module):
    """Small MLP: a Dense/relu stack, then an unactivated final
    ``Dense(output_dim)``. The network behind every tail: projection and
    visualization tails use it with a 2-3 or ``projection_dim``-wide output,
    classification tails with ``n_classes`` logits.

    ``hidden_dim`` is a sequence of layer widths, or an int for a single
    hidden layer.
    """

    hidden_dim: Union[int, Sequence[int]]
    output_dim: int

    @nn.compact
    def __call__(self, r: Array) -> Array:
        h = r
        for dim in _as_dims(self.hidden_dim):
            h = nn.relu(nn.Dense(dim)(h))
        return nn.Dense(self.output_dim)(h)


class _MLPTail:
    """Shared wrapper behind every tail: builds an ``MLP``, initializes its
    parameters (a mutable ``params`` pytree) and exposes the ``*_with_params``
    call used by training loops that differentiate with respect to ``params``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Union[int, Sequence[int]],
        output_dim: int,
        seed: int = 42,
    ):
        hidden_dims = _as_dims(hidden_dim)
        if input_dim <= 0 or not hidden_dims or any(d <= 0 for d in hidden_dims):
            raise ValueError("All dimensional arguments must be positive integers")
        if output_dim <= 0:
            raise ValueError("All dimensional arguments must be positive integers")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.module = MLP(hidden_dim=hidden_dim, output_dim=output_dim)
        init_r = jnp.zeros((1, input_dim), dtype=jnp.float32)
        rng = jax.random.PRNGKey(seed)
        variables = self.module.init({"params": rng}, init_r)
        self.params = variables["params"]

    def _apply_with_params(self, params: Params, r: Array) -> Array:
        return self.module.apply({"params": params}, jnp.asarray(r, dtype=jnp.float32))


class ProjectionTail(_MLPTail):
    """The projection tail: maps a SupCon body's representation ``r`` into
    the space the contrastive loss is actually computed in during phase-1
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
        super().__init__(input_dim, hidden_dim, projection_dim, seed)
        self.projection_dim = projection_dim

    def project_with_params(self, params: Params, r: Array) -> Array:
        """Project representation ``r`` using an explicit parameter tree."""
        return self._apply_with_params(params, r)

    def project(self, r: Array) -> Array:
        """Project representation ``r`` using the internally stored parameters."""
        return self.project_with_params(self.params, r)


class VisualizationTail(_MLPTail):
    """The visualization tail: maps a frozen SupCon body's representation
    ``r`` down to 2 or 3 dimensions for plotting, trained with the same
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
        if output_dim not in (2, 3):
            raise ValueError(f"output_dim must be 2 or 3, got {output_dim!r}")
        super().__init__(input_dim, hidden_dim, output_dim, seed)

    def project_with_params(self, params: Params, r: Array) -> Array:
        """Project representation ``r`` using an explicit parameter tree."""
        return self._apply_with_params(params, r)

    def project(self, r: Array) -> Array:
        """Project representation ``r`` using the internally stored parameters."""
        return self.project_with_params(self.params, r)


class ClassificationTail(_MLPTail):
    """A classifier on a frozen representation ``r``: an ``MLP`` with
    ``n_classes`` logits. The same class serves the family-level classifier
    (``n_classes`` = number of families) and a per-family spacegroup expert
    (``n_classes`` = number of spacegroups observed in that family).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Union[int, Sequence[int]],
        n_classes: int,
        seed: int = 42,
    ):
        """Initialize module architecture and random parameters.

        Args:
            input_dim: Width of the (frozen) representation ``r``.
            hidden_dim: Hidden width -- an int for a single hidden layer, or a
                sequence for a deeper MLP (one layer per entry, in order).
            n_classes: Number of output classes.
            seed: Random seed used for Flax parameter initialization.

        Raises:
            ValueError: If any dimensional argument is non-positive (every
                entry, when ``hidden_dim`` is a sequence -- which must also
                be non-empty).
        """
        super().__init__(input_dim, hidden_dim, n_classes, seed)
        self.n_classes = n_classes

    def classify_with_params(self, params: Params, r: Array) -> Array:
        """Return raw class logits for representation batch ``r``, using an
        explicit parameter tree."""
        return self._apply_with_params(params, r)

    def classify(self, r: Array) -> Array:
        """Return raw class logits using the stored parameters."""
        return self.classify_with_params(self.params, r)

    def load_params_bytes(self, data: bytes) -> None:
        """Load parameters saved with ``flax.serialization.to_bytes(tail.params)``.

        Also accepts the layout written before this tail had a single head,
        where the parameters sat under a ``"family_head"`` key
        (``{"family_head": {"Dense_0": ...}}``): that wrapper is unwrapped, so
        tails trained by older runs keep loading.
        """
        state = serialization.msgpack_restore(data)
        if "family_head" in state:
            state = state["family_head"]
        self.params = serialization.from_state_dict(self.params, state)
