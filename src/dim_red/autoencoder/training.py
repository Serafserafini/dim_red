"""
Training utilities for classic (non-variational) Autoencoder models.

Mirrors ``dim_red.vae.training``'s structure and public API (``TrainConfig``,
``train_autoencoder`` taking the same family/spacegroup id arguments as
``train_vae``) so the two are interchangeable from the pipeline's point of
view. The real difference is the objective: no KL term, no ``beta``, and no
PRNG key needed during training since encoding is deterministic.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from learned_optimization.research.general_lopt import prefab

from dim_red.autoencoder.model import Autoencoder, apply_family_mask
from dim_red.vae.database import VAEDatabase

Array = jax.Array


@dataclass(frozen=True)
class TrainConfig:
    """Training configuration for Autoencoder optimization.

    Attributes:
        epochs: Number of full passes over training data.
        batch_size: Number of samples per mini-batch.
        learning_rate: Kept for API compatibility; VeLO is still used as
            optimizer backend.
        lambda_family: Weight applied to the family classification
            cross-entropy term, when family ids are passed to
            ``train_autoencoder``. Ignored otherwise.
        lambda_spacegroup: Weight applied to the (family-masked) spacegroup
            classification cross-entropy term, when spacegroup ids are
            passed to ``train_autoencoder``. Ignored otherwise.
        seed: Seed controlling batch shuffling.
        device: JAX backend string (for example ``"cpu"`` or ``"gpu"``).
    """

    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    lambda_family: float = 0.0
    lambda_spacegroup: float = 0.0
    seed: int = 42
    device: str = "cpu"


def autoencoder_loss(x_recon: Array, x_true: Array) -> Array:
    """Reconstruction MSE -- the entire (unweighted) autoencoder objective.

    Args:
        x_recon: Reconstructed batch.
        x_true: Ground-truth batch.

    Returns:
        Scalar mean squared error between ``x_recon`` and ``x_true``.
    """
    return jnp.mean((x_recon - x_true) ** 2)


def _iter_batches(
    arrays: Tuple[np.ndarray, ...], batch_size: int, rng: np.random.Generator
) -> List[Tuple[np.ndarray, ...]]:
    """Shuffle a set of same-length arrays with one shared permutation and
    split them into aligned mini-batches.
    """
    n_samples = arrays[0].shape[0]
    indices = rng.permutation(n_samples)
    batches = []
    for start in range(0, n_samples, batch_size):
        batch_idx = indices[start : start + batch_size]
        batches.append(tuple(a[batch_idx] for a in arrays))
    return batches


def _make_train_step(
    model: Autoencoder,
    tx,
    lambda_family,
    lambda_spacegroup,
    has_family: bool,
    has_spacegroup: bool,
    family_spacegroup_mask,
):
    """Create a jitted training step bound to model, optimizer and loss weights.

    ``has_family``/``has_spacegroup`` are plain Python bools (not traced
    values): the branches they guard are resolved at trace time, so the
    unused-head code is compiled away entirely when a head is inactive.
    """

    @jax.jit
    def _train_step(params, batch_x, batch_family, batch_spacegroup, opt_state):
        """Single optimization step returning updated params/state/losses."""

        def loss_fn(local_params):
            z = model.encode_with_params(local_params, batch_x)
            x_recon = model.decode_with_params(local_params, z)
            recon = autoencoder_loss(x_recon, batch_x)
            total = recon

            family_ce = jnp.asarray(0.0)
            spacegroup_ce = jnp.asarray(0.0)
            if has_family:
                family_logits = model.classify_family_with_params(local_params, z)
                family_ce = optax.softmax_cross_entropy_with_integer_labels(
                    family_logits, batch_family
                ).mean()
                total = total + lambda_family * family_ce
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

            return total, (recon, family_ce, spacegroup_ce)

        (loss, (recon, family_ce, spacegroup_ce)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)
        updates, new_opt_state = tx.update(
            grads,
            opt_state,
            params,
            extra_args={"loss": loss},
        )
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, recon, family_ce, spacegroup_ce

    return _train_step


def _make_eval_step(
    model: Autoencoder,
    lambda_family,
    lambda_spacegroup,
    has_family: bool,
    has_spacegroup: bool,
    family_spacegroup_mask,
):
    """Create a jitted evaluation step returning per-sample losses."""

    @jax.jit
    def _eval_step(params, batch_x, batch_family, batch_spacegroup):
        """Compute per-sample validation losses for one batch."""
        z = model.encode_with_params(params, batch_x)
        x_recon = model.decode_with_params(params, z)
        recon = jnp.mean((x_recon - batch_x) ** 2, axis=1)
        total = recon

        family_ce = jnp.zeros_like(recon)
        spacegroup_ce = jnp.zeros_like(recon)
        if has_family:
            family_logits = model.classify_family_with_params(params, z)
            family_ce = optax.softmax_cross_entropy_with_integer_labels(
                family_logits, batch_family
            )
            total = total + lambda_family * family_ce
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

        return total, recon, family_ce, spacegroup_ce

    return _eval_step


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


def _make_eval_epoch(eval_step):
    """Create a jitted epoch-level parallel evaluator over mini-batches."""

    @jax.jit
    def _eval_epoch(
        params, batched_x, batched_family, batched_spacegroup, batched_mask
    ):
        total, recon, family_ce, spacegroup_ce = jax.vmap(
            eval_step, in_axes=(None, 0, 0, 0)
        )(params, batched_x, batched_family, batched_spacegroup)
        valid_count = jnp.maximum(jnp.sum(batched_mask), 1.0)

        def masked_mean(values):
            return jnp.sum(values * batched_mask) / valid_count

        return (
            masked_mean(total),
            masked_mean(recon),
            masked_mean(family_ce),
            masked_mean(spacegroup_ce),
        )

    return _eval_epoch


def train_autoencoder(
    model: Autoencoder,
    train_db: VAEDatabase,
    val_db: VAEDatabase,
    config: TrainConfig,
    train_family_ids: Optional[np.ndarray] = None,
    val_family_ids: Optional[np.ndarray] = None,
    train_spacegroup_ids: Optional[np.ndarray] = None,
    val_spacegroup_ids: Optional[np.ndarray] = None,
    family_spacegroup_mask: Optional[np.ndarray] = None,
) -> Dict[str, List[float]]:
    """Train an Autoencoder with VeLO (Optax wrapper) and return loss history.

    Args:
        model: Autoencoder instance containing Flax module and mutable parameters.
        train_db: Training dataset wrapper.
        val_db: Validation dataset wrapper.
        config: Training hyperparameters and execution backend options.
        train_family_ids: Integer family class ids (shape ``(n_train,)``),
            aligned row-for-row with ``train_db``. Together with
            ``val_family_ids`` and a model built with a family head,
            activates the family auxiliary loss (weighted by
            ``config.lambda_family``). Leave both ``None`` for a plain
            autoencoder.
        val_family_ids: Integer family class ids aligned with ``val_db``.
        train_spacegroup_ids: Integer spacegroup class ids (shape
            ``(n_train,)``), aligned with ``train_db``. Requires
            ``train_family_ids``/``val_family_ids`` and
            ``family_spacegroup_mask`` to also be given, and a model built
            with a spacegroup head. Activates the family-masked spacegroup
            auxiliary loss (weighted by ``config.lambda_spacegroup``).
        val_spacegroup_ids: Integer spacegroup class ids aligned with ``val_db``.
        family_spacegroup_mask: ``1.0``/``0.0`` co-occurrence matrix of
            shape ``(n_family_classes, n_spacegroup_classes)``, used to mask
            implausible spacegroups given the (here, true) family. See
            ``dim_red.autoencoder.model.apply_family_mask``.

    Returns:
        Dictionary with per-epoch losses. Always contains ``"train_loss"``,
        ``"train_recon"``, ``"val_loss"`` and ``"val_recon"`` (``train_loss``
        equals ``train_recon`` unless auxiliary heads are active, in which
        case ``*_loss`` is the full weighted objective actually optimized).
        Also contains ``"train_family_ce"``/``"val_family_ce"`` when family
        ids are given, and ``"train_spacegroup_ce"``/``"val_spacegroup_ce"``
        when spacegroup ids are given.

    Raises:
        ValueError: If config values are invalid, no matching JAX device is
            found, or the family/spacegroup id arguments are inconsistent
            (e.g. spacegroup ids given without family ids/mask).
    """
    if config.epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")
    if config.lambda_family < 0:
        raise ValueError("lambda_family must be >= 0")
    if config.lambda_spacegroup < 0:
        raise ValueError("lambda_spacegroup must be >= 0")

    has_family = train_family_ids is not None
    has_spacegroup = train_spacegroup_ids is not None
    if has_family != (val_family_ids is not None):
        raise ValueError("train_family_ids and val_family_ids must be given together")
    if has_spacegroup != (val_spacegroup_ids is not None):
        raise ValueError(
            "train_spacegroup_ids and val_spacegroup_ids must be given together"
        )
    if has_spacegroup and not (has_family and family_spacegroup_mask is not None):
        raise ValueError(
            "train_spacegroup_ids/val_spacegroup_ids require train_family_ids/"
            "val_family_ids and family_spacegroup_mask to also be provided "
            "(the spacegroup head is conditioned on family)"
        )

    devices = jax.devices(config.device)
    if not devices:
        raise ValueError(f"No JAX devices found for backend '{config.device}'")
    device = devices[0]

    train_np = np.asarray(train_db.data, dtype=np.float32)
    val_np = np.asarray(val_db.data, dtype=np.float32)

    train_family_np = (
        np.asarray(train_family_ids, dtype=np.int32)
        if has_family
        else np.zeros(train_np.shape[0], dtype=np.int32)
    )
    val_family_np = (
        np.asarray(val_family_ids, dtype=np.int32)
        if has_family
        else np.zeros(val_np.shape[0], dtype=np.int32)
    )
    train_spacegroup_np = (
        np.asarray(train_spacegroup_ids, dtype=np.int32)
        if has_spacegroup
        else np.zeros(train_np.shape[0], dtype=np.int32)
    )
    val_spacegroup_np = (
        np.asarray(val_spacegroup_ids, dtype=np.int32)
        if has_spacegroup
        else np.zeros(val_np.shape[0], dtype=np.int32)
    )
    mask_jax = (
        jnp.asarray(family_spacegroup_mask, dtype=jnp.float32)
        if has_spacegroup
        else jnp.zeros((1, 1), dtype=jnp.float32)
    )

    rng = np.random.default_rng(config.seed)
    batches_per_epoch = int(np.ceil(train_np.shape[0] / config.batch_size))
    total_steps = max(1, config.epochs * batches_per_epoch)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_recon": [],
        "val_loss": [],
        "val_recon": [],
    }
    if has_family:
        history["train_family_ce"] = []
        history["val_family_ce"] = []
    if has_spacegroup:
        history["train_spacegroup_ce"] = []
        history["val_spacegroup_ce"] = []

    tx = prefab.optax_lopt(num_steps=total_steps)
    lambda_family = jnp.asarray(config.lambda_family, dtype=jnp.float32)
    lambda_spacegroup = jnp.asarray(config.lambda_spacegroup, dtype=jnp.float32)

    train_step = _make_train_step(
        model,
        tx,
        lambda_family,
        lambda_spacegroup,
        has_family,
        has_spacegroup,
        mask_jax,
    )
    eval_step = _make_eval_step(
        model, lambda_family, lambda_spacegroup, has_family, has_spacegroup, mask_jax
    )
    eval_epoch = _make_eval_epoch(eval_step)
    opt_state = tx.init(model.params)
    params = model.params

    for _ in range(config.epochs):
        train_losses, train_recons = [], []
        train_family_ces, train_spacegroup_ces = [], []
        train_batches = _iter_batches(
            (train_np, train_family_np, train_spacegroup_np), config.batch_size, rng
        )
        for batch_x, batch_family, batch_spacegroup in train_batches:
            batch_x_jax = jax.device_put(jnp.asarray(batch_x), device)
            batch_family_jax = jax.device_put(jnp.asarray(batch_family), device)
            batch_spacegroup_jax = jax.device_put(jnp.asarray(batch_spacegroup), device)
            params, opt_state, loss, recon, family_ce, spacegroup_ce = train_step(
                params, batch_x_jax, batch_family_jax, batch_spacegroup_jax, opt_state
            )
            train_losses.append(float(loss))
            train_recons.append(float(recon))
            if has_family:
                train_family_ces.append(float(family_ce))
            if has_spacegroup:
                train_spacegroup_ces.append(float(spacegroup_ce))

        (
            val_batched_x,
            val_batched_family,
            val_batched_spacegroup,
        ), val_batched_mask = _prepare_padded_batches(
            (val_np, val_family_np, val_spacegroup_np), config.batch_size, rng
        )
        val_batched_x_jax = jax.device_put(jnp.asarray(val_batched_x), device)
        val_batched_family_jax = jax.device_put(jnp.asarray(val_batched_family), device)
        val_batched_spacegroup_jax = jax.device_put(
            jnp.asarray(val_batched_spacegroup), device
        )
        val_batched_mask_jax = jax.device_put(jnp.asarray(val_batched_mask), device)
        val_loss, val_recon, val_family_ce, val_spacegroup_ce = eval_epoch(
            params,
            val_batched_x_jax,
            val_batched_family_jax,
            val_batched_spacegroup_jax,
            val_batched_mask_jax,
        )

        history["train_loss"].append(float(np.mean(train_losses)))
        history["train_recon"].append(float(np.mean(train_recons)))
        history["val_loss"].append(float(val_loss))
        history["val_recon"].append(float(val_recon))
        if has_family:
            history["train_family_ce"].append(float(np.mean(train_family_ces)))
            history["val_family_ce"].append(float(val_family_ce))
        if has_spacegroup:
            history["train_spacegroup_ce"].append(float(np.mean(train_spacegroup_ces)))
            history["val_spacegroup_ce"].append(float(val_spacegroup_ce))

    model.params = params
    return history
