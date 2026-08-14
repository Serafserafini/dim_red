"""
CGCNN (Xie & Grossman 2018) model definitions using Flax Linen -- a
hand-written JAX/Flax reimplementation of the graph-convolutional encoder,
since no torch/torch_geometric/dgl/jraph is available in this project's
environment (see CLAUDE.md).

Two deliberate deviations from the original paper (both explained in
CLAUDE.md and ``dim_red.cgcnn.graph``):

- Atom features come from a learnable ``nn.Embed`` keyed by a per-structure
  LOCAL species slot (``dim_red.cgcnn.graph._local_species_indices``), not a
  real atomic number or a fixed lookup table -- this model must distinguish
  species A from species B within a structure without ever learning which
  real chemical element either one is.
- The original's two ``BatchNorm1d``s are replaced with ``nn.LayerNorm``:
  real batch-statistics BatchNorm needs mutable ``batch_stats``, a
  ``train``/``eval`` flag threaded through every call, and ``nn.vmap`` with
  an ``axis_name`` for correct cross-sample statistics -- machinery no other
  model wrapper in this codebase (``VAE``/``Autoencoder``/``SupConEncoder``)
  has, all exposing a single pure ``params`` pytree instead. ``LayerNorm``
  (per-sample, no mutable state) preserves the gating/residual formula while
  keeping ``CGCNNEncoder`` exactly as simple as its siblings.

Mirrors ``dim_red.autoencoder.model``'s public API shape
(``encode``/``classify_family``/``classify_spacegroup``, ``*_with_params``
variants, optional family/spacegroup auxiliary heads) so it slots into
``dim_red.pipeline.single_run``/``dim_red.pipeline.inference`` the same way
-- the only structural difference is that ``encode``/``encode_with_params``
take a 5-array graph batch instead of a single flat ``(batch, features)``
array. ``ClassifierHead``/``apply_family_mask`` are verbatim duplicates of
``dim_red.autoencoder.model``'s (not imported), matching this codebase's
no-cross-package-dependency convention (``dim_red.supcon`` does the same).
"""

from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

Array = jax.Array
Params = dict


class ConvLayer(nn.Module):
    """Gated graph-convolution layer (Xie & Grossman 2018's ``ConvLayer``),
    operating on a whole batch of graphs at once with an explicit leading
    batch dimension -- no ``jax.vmap`` needed, since the neighbor gather is
    plain batched advanced indexing.
    """

    atom_fea_len: int
    nbr_fea_len: int

    @nn.compact
    def __call__(
        self,
        atom_fea: Array,
        nbr_fea: Array,
        nbr_idx: Array,
        nbr_mask: Array,
        atom_mask: Array,
    ) -> Array:
        """One gated-convolution update of every atom's features.

        Args:
            atom_fea: ``(batch, max_atoms, atom_fea_len)``.
            nbr_fea: ``(batch, max_atoms, max_num_nbr, nbr_fea_len)``.
            nbr_idx: ``(batch, max_atoms, max_num_nbr)`` int, indexes the
                ``max_atoms`` axis of ``atom_fea`` within the same sample.
            nbr_mask: ``(batch, max_atoms, max_num_nbr)`` -- ``1.0`` for a
                real neighbor, ``0.0`` for a padded slot.
            atom_mask: ``(batch, max_atoms)`` -- ``1.0`` for a real atom,
                ``0.0`` for a padded slot.

        Returns:
            Updated atom features, same shape as ``atom_fea``.
        """
        batch = atom_fea.shape[0]
        batch_idx = jnp.arange(batch)[:, None, None]
        # Batched gather: atom_nbr_fea[b, a, k, :] = atom_fea[b, nbr_idx[b, a, k], :].
        atom_nbr_fea = atom_fea[batch_idx, nbr_idx]
        atom_self_fea = jnp.broadcast_to(atom_fea[:, :, None, :], atom_nbr_fea.shape)

        total_fea = jnp.concatenate([atom_self_fea, atom_nbr_fea, nbr_fea], axis=-1)
        gated = nn.Dense(2 * self.atom_fea_len)(total_fea)
        gated = nn.LayerNorm()(gated)
        filter_gate, core = jnp.split(gated, 2, axis=-1)
        nbr_msg = jax.nn.sigmoid(filter_gate) * jax.nn.softplus(core)
        nbr_msg = nbr_msg * nbr_mask[..., None]

        nbr_sum = jnp.sum(nbr_msg, axis=2)
        nbr_sum = nn.LayerNorm()(nbr_sum)
        out = jax.nn.softplus(atom_fea + nbr_sum)
        return out * atom_mask[..., None]


class CGCNNBodyModule(nn.Module):
    """Embedding -> stacked :class:`ConvLayer`\\ s -> masked mean pooling ->
    post-pooling MLP -> final ``latent_dim`` output. Plays the role of
    ``EncoderModule`` in ``vae``/``autoencoder``/``supcon`` -- graph input
    instead of a flat feature vector.
    """

    atom_fea_len: int
    n_conv: int
    h_fea_len: int
    n_h: int
    latent_dim: int
    n_gaussian: int
    max_species: int

    @nn.compact
    def __call__(
        self,
        local_species_idx: Array,
        nbr_idx: Array,
        nbr_fea: Array,
        nbr_mask: Array,
        atom_mask: Array,
    ) -> Array:
        """Encode a batch of padded crystal graphs into ``(batch, latent_dim)``.

        Args:
            local_species_idx: ``(batch, max_atoms)`` int -- per-structure
                LOCAL species slot (``0`` = padding), not a real atomic
                number. See module docstring.
            nbr_idx: ``(batch, max_atoms, max_num_nbr)`` int.
            nbr_fea: ``(batch, max_atoms, max_num_nbr, n_gaussian)``.
            nbr_mask: ``(batch, max_atoms, max_num_nbr)``.
            atom_mask: ``(batch, max_atoms)``.

        Returns:
            ``(batch, latent_dim)`` crystal-level representation.
        """
        # +1 embedding row reserves index 0 for the padding sentinel.
        atom_fea = nn.Embed(
            num_embeddings=self.max_species + 1, features=self.atom_fea_len
        )(local_species_idx)

        for _ in range(self.n_conv):
            atom_fea = ConvLayer(
                atom_fea_len=self.atom_fea_len, nbr_fea_len=self.n_gaussian
            )(atom_fea, nbr_fea, nbr_idx, nbr_mask, atom_mask)

        summed = jnp.sum(atom_fea * atom_mask[..., None], axis=1)
        counts = jnp.maximum(jnp.sum(atom_mask, axis=1, keepdims=True), 1.0)
        pooled = summed / counts

        h = pooled
        for _ in range(self.n_h):
            h = jax.nn.softplus(nn.Dense(self.h_fea_len)(h))
        return nn.Dense(self.latent_dim)(h)


class ClassifierHead(nn.Module):
    """Small MLP classifying representations into discrete classes.

    Verbatim duplicate of ``dim_red.autoencoder.model.ClassifierHead``.
    """

    hidden_dim: int
    n_classes: int

    @nn.compact
    def __call__(self, z: Array) -> Array:
        """Return class logits for a batch of representations ``z``."""
        h = nn.relu(nn.Dense(self.hidden_dim)(z))
        return nn.Dense(self.n_classes)(h)


def apply_family_mask(
    spacegroup_logits: Array,
    family_weights: Array,
    family_spacegroup_mask: Array,
    eps: float = 1e-8,
) -> Array:
    """Condition spacegroup logits on a family weighting via additive log-space masking.

    Verbatim duplicate of ``dim_red.autoencoder.model.apply_family_mask``
    (and ``dim_red.vae.model``'s) -- kept independent so this package has no
    dependency on ``vae``/``autoencoder``/``supcon``.
    """
    plausibility = family_weights @ family_spacegroup_mask
    return spacegroup_logits + jnp.log(plausibility + eps)


class CGCNNEncoderModule(nn.Module):
    """Body + optional family/spacegroup classifier heads on the body's
    output. Mirrors ``AutoencoderModule``'s ``setup()``/``classify_family``/
    ``classify_spacegroup``/``init_all`` shape, minus ``decode`` -- this
    model has no decoder at all, like ``SupConEncoder``.
    """

    atom_fea_len: int
    n_conv: int
    h_fea_len: int
    n_h: int
    latent_dim: int
    n_gaussian: int
    max_species: int
    n_family_classes: Optional[int] = None
    n_spacegroup_classes: Optional[int] = None
    head_hidden_dim: int = 16

    def setup(self):
        """Initialize the body and any configured classifier heads."""
        self.body = CGCNNBodyModule(
            atom_fea_len=self.atom_fea_len,
            n_conv=self.n_conv,
            h_fea_len=self.h_fea_len,
            n_h=self.n_h,
            latent_dim=self.latent_dim,
            n_gaussian=self.n_gaussian,
            max_species=self.max_species,
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

    def encode(
        self,
        local_species_idx: Array,
        nbr_idx: Array,
        nbr_fea: Array,
        nbr_mask: Array,
        atom_mask: Array,
    ) -> Array:
        """Run the body and return the deterministic representation ``z``."""
        return self.body(local_species_idx, nbr_idx, nbr_fea, nbr_mask, atom_mask)

    def classify_family(self, z: Array) -> Array:
        """Return raw (unmasked) family logits for representation batch ``z``."""
        if self.family_head is None:
            raise ValueError("Family head is not configured (n_family_classes is None)")
        return self.family_head(z)

    def classify_spacegroup(self, z: Array) -> Array:
        """Return raw (unmasked) spacegroup logits for representation batch ``z``.

        Callers are expected to condition these on family via
        :func:`apply_family_mask`.
        """
        if self.spacegroup_head is None:
            raise ValueError(
                "Spacegroup head is not configured (n_spacegroup_classes is None)"
            )
        return self.spacegroup_head(z)

    def __call__(
        self,
        local_species_idx: Array,
        nbr_idx: Array,
        nbr_fea: Array,
        nbr_mask: Array,
        atom_mask: Array,
    ) -> Array:
        """Alias for :meth:`encode` -- there is no reconstruction target."""
        return self.encode(local_species_idx, nbr_idx, nbr_fea, nbr_mask, atom_mask)

    def init_all(
        self,
        local_species_idx: Array,
        nbr_idx: Array,
        nbr_fea: Array,
        nbr_mask: Array,
        atom_mask: Array,
    ) -> Array:
        """Trace the body and every configured head, for parameter allocation.

        Used only as the ``method`` passed to ``Module.init`` -- ``__call__``
        itself never touches the heads, since it must keep returning just
        ``z`` regardless of which heads are configured.
        """
        z = self.encode(local_species_idx, nbr_idx, nbr_fea, nbr_mask, atom_mask)
        if self.family_head is not None:
            self.classify_family(z)
        if self.spacegroup_head is not None:
            self.classify_spacegroup(z)
        return z


class Encoder:
    """Thin adapter exposing only the encoder interface of a trained ``CGCNNEncoder``."""

    def __init__(self, cgcnn_encoder: "CGCNNEncoder"):
        self._cgcnn_encoder = cgcnn_encoder

    def __call__(self, graph_batch: Tuple) -> Array:
        """Return the representation ``z`` for a graph batch."""
        return self._cgcnn_encoder.encode(graph_batch)


class CGCNNEncoder:
    """High-level wrapper managing Flax module creation and parameters for
    the CGCNN body + optional classifier heads -- mirrors
    ``dim_red.autoencoder.model.Autoencoder``'s shape (mutable ``params``,
    ``encode``/``classify_family``/``classify_spacegroup``,
    ``*_with_params`` variants), except ``encode``/``encode_with_params``
    take a graph batch (a 5-array tuple) instead of a flat
    ``(batch, features)`` array, and there is no decoder at all.
    """

    def __init__(
        self,
        atom_fea_len: int = 64,
        n_conv: int = 3,
        h_fea_len: int = 128,
        n_h: int = 1,
        latent_dim: int = 32,
        n_gaussian: int = 41,
        max_species: int = 10,
        n_family_classes: Optional[int] = None,
        n_spacegroup_classes: Optional[int] = None,
        head_hidden_dim: int = 16,
        seed: int = 42,
    ):
        """Initialize module architecture and random model parameters.

        Args:
            atom_fea_len: Per-atom embedding/hidden width used throughout
                the graph-convolution stack.
            n_conv: Number of stacked gated graph-convolution layers.
            h_fea_len: Hidden width of the post-pooling fully-connected block.
            n_h: Number of post-pooling hidden layers.
            latent_dim: Size of the pooled crystal-level representation
                (the body's output width, analogous to ``latent_dim``
                elsewhere in this codebase).
            n_gaussian: Width of the Gaussian-expanded bond/edge feature
                vector fed into every :class:`ConvLayer` -- must match
                whatever produced the graph batch's ``nbr_fea`` (see
                ``dim_red.cgcnn.graph``).
            max_species: Maximum distinct LOCAL species slots any structure
                may use -- must match whatever produced the graph batch's
                ``local_species_idx``. Controls the embedding table size
                (``max_species + 1`` rows, the ``+1`` reserving slot ``0``
                for padding).
            n_family_classes: If set, adds a small family-classification
                head on the pooled representation. ``None`` (default) means
                no auxiliary heads at all.
            n_spacegroup_classes: If set, adds a small spacegroup-
                classification head, conditioned on family via
                :func:`apply_family_mask` (applied by the caller, not this
                class). Requires ``n_family_classes`` to also be set.
            head_hidden_dim: Hidden width of each auxiliary head's single
                hidden layer.
            seed: Random seed used for Flax parameter initialization.

        Raises:
            ValueError: If any dimensional argument is non-positive, or if
                ``n_spacegroup_classes`` is set without ``n_family_classes``.
        """
        if n_spacegroup_classes is not None and n_family_classes is None:
            raise ValueError(
                "n_family_classes must be set when n_spacegroup_classes is set "
                "(the spacegroup head is conditioned on family)"
            )
        if (
            atom_fea_len <= 0
            or n_conv <= 0
            or h_fea_len <= 0
            or n_h <= 0
            or latent_dim <= 0
            or n_gaussian <= 0
            or max_species <= 0
        ):
            raise ValueError("All dimensional arguments must be positive integers")

        self.atom_fea_len = atom_fea_len
        self.n_conv = n_conv
        self.h_fea_len = h_fea_len
        self.n_h = n_h
        self.latent_dim = latent_dim
        self.n_gaussian = n_gaussian
        self.max_species = max_species
        self.n_family_classes = n_family_classes
        self.n_spacegroup_classes = n_spacegroup_classes
        self.head_hidden_dim = head_hidden_dim

        self.module = CGCNNEncoderModule(
            atom_fea_len=atom_fea_len,
            n_conv=n_conv,
            h_fea_len=h_fea_len,
            n_h=n_h,
            latent_dim=latent_dim,
            n_gaussian=n_gaussian,
            max_species=max_species,
            n_family_classes=n_family_classes,
            n_spacegroup_classes=n_spacegroup_classes,
            head_hidden_dim=head_hidden_dim,
        )
        # Shapes don't matter for parameter allocation beyond n_gaussian, so
        # a single-atom, single-neighbor dummy graph suffices.
        init_local_species_idx = jnp.zeros((1, 1), dtype=jnp.int32)
        init_nbr_idx = jnp.zeros((1, 1, 1), dtype=jnp.int32)
        init_nbr_fea = jnp.zeros((1, 1, 1, n_gaussian), dtype=jnp.float32)
        init_nbr_mask = jnp.zeros((1, 1, 1), dtype=jnp.float32)
        init_atom_mask = jnp.ones((1, 1), dtype=jnp.float32)
        rng = jax.random.PRNGKey(seed)
        # `init_all` (rather than `__call__`) traces the classifier heads
        # too, if configured, so their params get allocated here regardless
        # of which heads are present -- see `CGCNNEncoderModule.init_all`.
        variables = self.module.init(
            {"params": rng},
            init_local_species_idx,
            init_nbr_idx,
            init_nbr_fea,
            init_nbr_mask,
            init_atom_mask,
            method=self.module.init_all,
        )
        self.params = variables["params"]
        self.encoder = Encoder(self)

    def encode_with_params(self, params: Params, graph_batch: Tuple) -> Array:
        """Encode a graph batch using an explicit parameter tree.

        Args:
            params: Parameter pytree.
            graph_batch: ``(local_species_idx, nbr_idx, nbr_fea, nbr_mask,
                atom_mask)``, each with a leading batch dimension -- see
                ``dim_red.cgcnn.graph.atoms_list_to_graph_arrays``.
        """
        local_species_idx, nbr_idx, nbr_fea, nbr_mask, atom_mask = graph_batch
        return self.module.apply(
            {"params": params},
            jnp.asarray(local_species_idx, dtype=jnp.int32),
            jnp.asarray(nbr_idx, dtype=jnp.int32),
            jnp.asarray(nbr_fea, dtype=jnp.float32),
            jnp.asarray(nbr_mask, dtype=jnp.float32),
            jnp.asarray(atom_mask, dtype=jnp.float32),
            method=self.module.encode,
        )

    def classify_family_with_params(self, params: Params, z: Array) -> Array:
        """Return raw (unmasked) family logits for representation batch ``z``.

        Raises:
            ValueError: If the model was built without a family head.
        """
        return self.module.apply(
            {"params": params},
            jnp.asarray(z, dtype=jnp.float32),
            method=self.module.classify_family,
        )

    def classify_spacegroup_with_params(self, params: Params, z: Array) -> Array:
        """Return raw (unmasked) spacegroup logits for representation batch ``z``.

        Raises:
            ValueError: If the model was built without a spacegroup head.
        """
        return self.module.apply(
            {"params": params},
            jnp.asarray(z, dtype=jnp.float32),
            method=self.module.classify_spacegroup,
        )

    def encode(self, graph_batch: Tuple) -> Array:
        """Encode a graph batch using the internally stored model parameters."""
        return self.encode_with_params(self.params, graph_batch)

    def classify_family(self, z: Array) -> Array:
        """Return raw (unmasked) family logits using the stored parameters."""
        return self.classify_family_with_params(self.params, z)

    def classify_spacegroup(self, z: Array) -> Array:
        """Return raw (unmasked) spacegroup logits using the stored parameters."""
        return self.classify_spacegroup_with_params(self.params, z)
