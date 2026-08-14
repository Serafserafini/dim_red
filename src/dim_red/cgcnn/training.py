"""
Training utilities for CGCNN -- a single-phase loop that jointly trains the
body and classifier head(s) via cross-entropy on family (+ optional
family-masked spacegroup) labels.

Mirrors ``dim_red.autoencoder.training``'s aux-heads training pattern (not
``dim_red.supcon.training``'s two-phase contrastive design): there is no
reconstruction term, no KL, no contrastive loss -- classification is
CGCNN's only training objective. ``train_family_ids``/``val_family_ids`` are
therefore REQUIRED (not ``Optional``, unlike
``autoencoder.training.train_autoencoder``), since a family-less call would
be a pure no-op that silently trains nothing.

``_iter_batches``/``_make_optimizer``/``_prepare_padded_batches``/
``_make_eval_epoch`` are verbatim duplicates of
``dim_red.autoencoder.training``'s -- already rank-agnostic (they operate
generically on any tuple of same-leading-dim arrays), so they work unchanged
on the five graph arrays (2D/3D/4D) a ``GraphDatabase`` holds, alongside the
1D family/spacegroup id arrays.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from learned_optimization.research.general_lopt import prefab

from dim_red.cgcnn.database import GraphDatabase
from dim_red.cgcnn.model import CGCNNEncoder, apply_family_mask

Array = jax.Array

_OPTIMIZERS = ("adam", "velo")


@dataclass(frozen=True)
class TrainConfig:
    """Training configuration for ``CGCNNEncoder`` optimization.

    Attributes:
        epochs: Number of full passes over training data.
        batch_size: Number of samples per mini-batch.
        learning_rate: Adam's learning rate, used whenever ``optimizer ==
            "adam"`` (the default). Ignored (kept for API compatibility) when
            ``optimizer == "velo"``.
        optimizer: ``"adam"`` (default) -- a plain ``optax.adam(learning_rate)``
            -- or ``"velo"`` -- ``learned_optimization``'s pretrained VeLO
            meta-learned optimizer, whose ``num_steps``-dependent setup and
            pretrained-hypernetwork checkpoint load cost real, fixed time
            before training even starts. See ``_make_optimizer``.
        lambda_family: Weight applied to the (always-active) family
            classification cross-entropy term.
        lambda_spacegroup: Weight applied to the (family-masked) spacegroup
            classification cross-entropy term, when spacegroup ids are
            passed to ``train_cgcnn``. Ignored otherwise.
        seed: Seed controlling batch shuffling.
        device: JAX backend string (for example ``"cpu"`` or ``"gpu"``).
        early_stopping: If True, stop training once ``val_loss`` (the full
            weighted objective) hasn't improved by more than
            ``early_stopping_min_delta`` for ``early_stopping_patience``
            consecutive epochs. Disabled by default -- the monitored metric
            is always ``val_loss``, not configurable.
        early_stopping_patience: Consecutive non-improving epochs tolerated
            before stopping. Ignored unless ``early_stopping`` is True.
        early_stopping_min_delta: Minimum decrease in ``val_loss`` counted
            as an improvement. Ignored unless ``early_stopping`` is True.
        early_stopping_restore_best: If True (default), the returned
            ``model.params`` are the best-``val_loss`` epoch's rather than
            necessarily the last epoch trained. Ignored unless
            ``early_stopping`` is True.
    """

    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    lambda_family: float = 1.0
    lambda_spacegroup: float = 1.0
    seed: int = 42
    device: str = "cpu"
    early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0
    early_stopping_restore_best: bool = True


def _iter_batches(
    arrays: Tuple[np.ndarray, ...], batch_size: int, rng: np.random.Generator
) -> List[Tuple[np.ndarray, ...]]:
    """Shuffle a set of same-length arrays with one shared permutation and
    split them into aligned mini-batches (last batch may be smaller).
    """
    n_samples = arrays[0].shape[0]
    indices = rng.permutation(n_samples)
    batches = []
    for start in range(0, n_samples, batch_size):
        batch_idx = indices[start : start + batch_size]
        batches.append(tuple(a[batch_idx] for a in arrays))
    return batches


def _make_optimizer(optimizer: str, learning_rate: float, num_steps: int):
    """Build this training loop's optax-compatible optimizer.

    ``"adam"`` (default) is a plain ``optax.adam(learning_rate)``, wrapped in
    ``optax.with_extra_args_support`` so it also accepts the
    ``extra_args={"loss": ...}`` kwarg every train/eval step always passes
    (VeLO is loss-conditioned; a bare ``optax.adam`` doesn't accept
    ``extra_args`` at all). ``"velo"`` uses ``learned_optimization``'s
    pretrained VeLO meta-learned optimizer instead.
    """
    if optimizer == "velo":
        return prefab.optax_lopt(num_steps=num_steps)
    return optax.with_extra_args_support(optax.adam(learning_rate))


def _prepare_padded_batches(
    arrays: Tuple[np.ndarray, ...], batch_size: int, rng: np.random.Generator
) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
    """Create fixed-size shuffled, aligned batches (one shared permutation)
    and a validity mask for padding.

    Returns:
        A tuple ``(batched_arrays, batched_mask)`` where:
        - ``batched_arrays[i]`` has shape ``(n_batches, batch_size, *arrays[i].shape[1:])``.
        - ``batched_mask`` has shape ``(n_batches, batch_size)`` with 1.0 on
          valid samples and 0.0 on padded entries.
    """
    n_samples = arrays[0].shape[0]
    indices = rng.permutation(n_samples)
    shuffled = [a[indices] for a in arrays]
    n_batches = int(np.ceil(n_samples / batch_size))
    total_slots = n_batches * batch_size
    pad = total_slots - n_samples

    if pad > 0:
        shuffled = [
            np.pad(a, [(0, pad)] + [(0, 0)] * (a.ndim - 1), mode="constant")
            for a in shuffled
        ]

    batched = tuple(a.reshape(n_batches, batch_size, *a.shape[1:]) for a in shuffled)
    batched_mask = np.ones((n_batches, batch_size), dtype=np.float32)
    if pad > 0:
        batched_mask[-1, batch_size - pad :] = 0.0
    return batched, batched_mask


def _make_train_step(
    model: CGCNNEncoder,
    tx,
    lambda_family,
    lambda_spacegroup,
    has_spacegroup: bool,
    family_spacegroup_mask,
):
    """Create a jitted training step bound to model, optimizer and loss weights.

    Unlike ``autoencoder.training``'s, ``has_family`` is not a parameter
    here -- family classification is unconditionally active (``train_cgcnn``
    requires ``train_family_ids``/``val_family_ids``, see below).
    """

    @jax.jit
    def _train_step(params, batch_graph, batch_family, batch_spacegroup, opt_state):
        """Single optimization step returning updated params/state/losses."""

        def loss_fn(local_params):
            z = model.encode_with_params(local_params, batch_graph)
            family_logits = model.classify_family_with_params(local_params, z)
            family_ce = optax.softmax_cross_entropy_with_integer_labels(
                family_logits, batch_family
            ).mean()
            total = lambda_family * family_ce

            spacegroup_ce = jnp.asarray(0.0)
            if has_spacegroup:
                spacegroup_logits = model.classify_spacegroup_with_params(
                    local_params, z
                )
                family_onehot = jax.nn.one_hot(
                    batch_family, family_spacegroup_mask.shape[0]
                )
                masked_logits = apply_family_mask(
                    spacegroup_logits, family_onehot, family_spacegroup_mask
                )
                spacegroup_ce = optax.softmax_cross_entropy_with_integer_labels(
                    masked_logits, batch_spacegroup
                ).mean()
                total = total + lambda_spacegroup * spacegroup_ce

            return total, (family_ce, spacegroup_ce)

        (loss, (family_ce, spacegroup_ce)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)
        updates, new_opt_state = tx.update(
            grads, opt_state, params, extra_args={"loss": loss}
        )
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, family_ce, spacegroup_ce

    return _train_step


def _make_eval_step(
    model: CGCNNEncoder,
    lambda_family,
    lambda_spacegroup,
    has_spacegroup: bool,
    family_spacegroup_mask,
):
    """Create a jitted evaluation step returning per-sample losses -- required
    for the padded+vmap-over-epoch trick below.
    """

    @jax.jit
    def _eval_step(params, batch_graph, batch_family, batch_spacegroup):
        """Compute per-sample validation losses for one batch."""
        z = model.encode_with_params(params, batch_graph)
        family_logits = model.classify_family_with_params(params, z)
        family_ce = optax.softmax_cross_entropy_with_integer_labels(
            family_logits, batch_family
        )
        total = lambda_family * family_ce

        spacegroup_ce = jnp.zeros_like(family_ce)
        if has_spacegroup:
            spacegroup_logits = model.classify_spacegroup_with_params(params, z)
            family_onehot = jax.nn.one_hot(
                batch_family, family_spacegroup_mask.shape[0]
            )
            masked_logits = apply_family_mask(
                spacegroup_logits, family_onehot, family_spacegroup_mask
            )
            spacegroup_ce = optax.softmax_cross_entropy_with_integer_labels(
                masked_logits, batch_spacegroup
            )
            total = total + lambda_spacegroup * spacegroup_ce

        return total, family_ce, spacegroup_ce

    return _eval_step


def _make_eval_epoch(eval_step):
    """Create a jitted epoch-level parallel evaluator over mini-batches.

    ``batched_graph`` is itself a 5-tuple of arrays each shaped
    ``(n_batches, batch_size, ...)`` -- ``jax.vmap``'s ``in_axes=0`` applied
    to a Python tuple pytree automatically maps axis 0 of every leaf array
    in that tuple, so ``eval_step`` receives a genuine
    ``(batch_size, ...)`` slice for each of the 5 graph arrays per vmap
    iteration, with no special-casing needed versus a single flat array.
    """

    @jax.jit
    def _eval_epoch(
        params, batched_graph, batched_family, batched_spacegroup, batched_mask
    ):
        total, family_ce, spacegroup_ce = jax.vmap(eval_step, in_axes=(None, 0, 0, 0))(
            params, batched_graph, batched_family, batched_spacegroup
        )
        valid_count = jnp.maximum(jnp.sum(batched_mask), 1.0)

        def masked_mean(values):
            return jnp.sum(values * batched_mask) / valid_count

        return masked_mean(total), masked_mean(family_ce), masked_mean(spacegroup_ce)

    return _eval_epoch


def train_cgcnn(
    model: CGCNNEncoder,
    train_db: GraphDatabase,
    val_db: GraphDatabase,
    config: TrainConfig,
    train_family_ids: np.ndarray,
    val_family_ids: np.ndarray,
    train_spacegroup_ids: Optional[np.ndarray] = None,
    val_spacegroup_ids: Optional[np.ndarray] = None,
    family_spacegroup_mask: Optional[np.ndarray] = None,
) -> Dict[str, List[float]]:
    """Train a ``CGCNNEncoder`` body + classifier head(s) jointly (Adam by
    default, or VeLO -- see ``TrainConfig.optimizer``) via cross-entropy on
    family (+ optional family-masked spacegroup) labels, and return
    per-epoch loss history. ``model.params`` is updated in place.

    Unlike ``train_vae``/``train_autoencoder``, ``train_family_ids``/
    ``val_family_ids`` are REQUIRED (not ``Optional``) -- CGCNN has no other
    training objective at all (no reconstruction, no contrastive loss), so a
    family-less call would be a pure no-op that silently trains nothing.

    Args:
        model: ``CGCNNEncoder`` instance containing the Flax module and
            mutable parameters.
        train_db: Training ``GraphDatabase``.
        val_db: Validation ``GraphDatabase``.
        config: Training hyperparameters and execution backend options.
        train_family_ids: Integer family class ids (shape ``(n_train,)``),
            aligned row-for-row with ``train_db``. Required.
        val_family_ids: Integer family class ids aligned with ``val_db``. Required.
        train_spacegroup_ids: Integer spacegroup class ids (shape
            ``(n_train,)``), aligned with ``train_db``. Requires
            ``val_spacegroup_ids`` and ``family_spacegroup_mask`` to also be
            given, and a model built with a spacegroup head. Activates the
            family-masked spacegroup auxiliary loss (weighted by
            ``config.lambda_spacegroup``).
        val_spacegroup_ids: Integer spacegroup class ids aligned with ``val_db``.
        family_spacegroup_mask: ``1.0``/``0.0`` co-occurrence matrix of
            shape ``(n_family_classes, n_spacegroup_classes)``. See
            ``dim_red.cgcnn.model.apply_family_mask``.

    Returns:
        Dictionary with per-epoch losses, one entry per epoch actually run
        (shorter than ``config.epochs`` if ``config.early_stopping`` stopped
        training early). Always contains ``"train_loss"``/``"train_family_ce"``/
        ``"val_loss"``/``"val_family_ce"`` (``train_loss``/``val_loss`` equal
        the full weighted objective, which is ``lambda_family * family_ce``
        alone unless the spacegroup head is also active). Also contains
        ``"train_spacegroup_ce"``/``"val_spacegroup_ce"`` when spacegroup
        ids are given.

    Raises:
        ValueError: If ``train_family_ids``/``val_family_ids`` is ``None``,
            config values are invalid, no matching JAX device is found, or
            the spacegroup id arguments are inconsistent (e.g. spacegroup
            ids given without ``family_spacegroup_mask``).
    """
    if train_family_ids is None or val_family_ids is None:
        raise ValueError(
            "train_family_ids and val_family_ids are required for train_cgcnn "
            "-- family classification is CGCNN's only training objective "
            "(no reconstruction/contrastive fallback exists for this model)"
        )
    if config.epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")
    if config.optimizer not in _OPTIMIZERS:
        raise ValueError(
            f"optimizer must be one of {_OPTIMIZERS}, got {config.optimizer!r}"
        )
    if config.lambda_family < 0:
        raise ValueError("lambda_family must be >= 0")
    if config.lambda_spacegroup < 0:
        raise ValueError("lambda_spacegroup must be >= 0")
    if config.early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be a positive integer")
    if config.early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta must be >= 0")

    has_spacegroup = train_spacegroup_ids is not None
    if has_spacegroup != (val_spacegroup_ids is not None):
        raise ValueError(
            "train_spacegroup_ids and val_spacegroup_ids must be given together"
        )
    if has_spacegroup and family_spacegroup_mask is None:
        raise ValueError(
            "train_spacegroup_ids/val_spacegroup_ids require "
            "family_spacegroup_mask to also be provided (the spacegroup "
            "head is conditioned on family)"
        )

    devices = jax.devices(config.device)
    if not devices:
        raise ValueError(f"No JAX devices found for backend '{config.device}'")
    device = devices[0]

    train_graph = train_db.as_tuple()
    val_graph = val_db.as_tuple()
    n_train = train_db.n_samples
    n_val = val_db.n_samples

    train_family_np = np.asarray(train_family_ids, dtype=np.int32)
    val_family_np = np.asarray(val_family_ids, dtype=np.int32)
    train_spacegroup_np = (
        np.asarray(train_spacegroup_ids, dtype=np.int32)
        if has_spacegroup
        else np.zeros(n_train, dtype=np.int32)
    )
    val_spacegroup_np = (
        np.asarray(val_spacegroup_ids, dtype=np.int32)
        if has_spacegroup
        else np.zeros(n_val, dtype=np.int32)
    )
    mask_jax = (
        jnp.asarray(family_spacegroup_mask, dtype=jnp.float32)
        if has_spacegroup
        else jnp.zeros((1, 1), dtype=jnp.float32)
    )

    rng = np.random.default_rng(config.seed)
    batches_per_epoch = int(np.ceil(n_train / config.batch_size))
    total_steps = max(1, config.epochs * batches_per_epoch)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_family_ce": [],
        "val_loss": [],
        "val_family_ce": [],
    }
    if has_spacegroup:
        history["train_spacegroup_ce"] = []
        history["val_spacegroup_ce"] = []

    tx = _make_optimizer(config.optimizer, config.learning_rate, total_steps)
    lambda_family = jnp.asarray(config.lambda_family, dtype=jnp.float32)
    lambda_spacegroup = jnp.asarray(config.lambda_spacegroup, dtype=jnp.float32)

    train_step = _make_train_step(
        model, tx, lambda_family, lambda_spacegroup, has_spacegroup, mask_jax
    )
    eval_step = _make_eval_step(
        model, lambda_family, lambda_spacegroup, has_spacegroup, mask_jax
    )
    eval_epoch = _make_eval_epoch(eval_step)
    opt_state = tx.init(model.params)
    params = model.params

    best_val_loss = float("inf")
    best_params = None
    epochs_without_improvement = 0

    for _ in range(config.epochs):
        train_losses, train_family_ces, train_spacegroup_ces = [], [], []
        train_batches = _iter_batches(
            train_graph + (train_family_np, train_spacegroup_np),
            config.batch_size,
            rng,
        )
        for batch in train_batches:
            batch_graph = batch[:5]
            batch_family, batch_spacegroup = batch[5], batch[6]
            batch_graph_jax = tuple(
                jax.device_put(jnp.asarray(a), device) for a in batch_graph
            )
            batch_family_jax = jax.device_put(jnp.asarray(batch_family), device)
            batch_spacegroup_jax = jax.device_put(jnp.asarray(batch_spacegroup), device)
            params, opt_state, loss, family_ce, spacegroup_ce = train_step(
                params,
                batch_graph_jax,
                batch_family_jax,
                batch_spacegroup_jax,
                opt_state,
            )
            train_losses.append(float(loss))
            train_family_ces.append(float(family_ce))
            if has_spacegroup:
                train_spacegroup_ces.append(float(spacegroup_ce))

        val_batched, val_batched_mask = _prepare_padded_batches(
            val_graph + (val_family_np, val_spacegroup_np), config.batch_size, rng
        )
        val_batched_graph = val_batched[:5]
        val_batched_family, val_batched_spacegroup = val_batched[5], val_batched[6]
        val_batched_graph_jax = tuple(
            jax.device_put(jnp.asarray(a), device) for a in val_batched_graph
        )
        val_batched_family_jax = jax.device_put(jnp.asarray(val_batched_family), device)
        val_batched_spacegroup_jax = jax.device_put(
            jnp.asarray(val_batched_spacegroup), device
        )
        val_batched_mask_jax = jax.device_put(jnp.asarray(val_batched_mask), device)
        val_loss, val_family_ce, val_spacegroup_ce = eval_epoch(
            params,
            val_batched_graph_jax,
            val_batched_family_jax,
            val_batched_spacegroup_jax,
            val_batched_mask_jax,
        )

        history["train_loss"].append(float(np.mean(train_losses)))
        history["train_family_ce"].append(float(np.mean(train_family_ces)))
        history["val_loss"].append(float(val_loss))
        history["val_family_ce"].append(float(val_family_ce))
        if has_spacegroup:
            history["train_spacegroup_ce"].append(float(np.mean(train_spacegroup_ces)))
            history["val_spacegroup_ce"].append(float(val_spacegroup_ce))

        if config.early_stopping:
            current_val_loss = history["val_loss"][-1]
            if current_val_loss < best_val_loss - config.early_stopping_min_delta:
                best_val_loss = current_val_loss
                epochs_without_improvement = 0
                if config.early_stopping_restore_best:
                    best_params = params
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.early_stopping_patience:
                    break

    if (
        config.early_stopping
        and config.early_stopping_restore_best
        and best_params is not None
    ):
        params = best_params
    model.params = params
    return history
