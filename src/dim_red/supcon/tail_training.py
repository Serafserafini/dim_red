"""
Phase 2 training: freeze an already-trained SupCon body and train exactly
one tail (classification or visualization) on its precomputed
representations -- see ``dim_red.supcon.training`` for phase 1 (body +
projection tail, jointly trained).

Both ``train_classification_tail``/``train_visualization_tail`` take plain
``r_train``/``r_val`` arrays (the body's already-computed, already-frozen
representations -- e.g. a completed run's ``embeddings.npz["embeddings"]``,
see ``dim_red.pipeline.tail_training``), not a ``VAEDatabase``: these are
fixed representations to train a small head on, not raw features to
standardize/encode. The body's own parameters never appear in either
training loop's ``value_and_grad`` call at all -- there is nothing to
"freeze" beyond simply not including them, which is why neither function
needs ``jax.lax.stop_gradient``/masked-optimizer machinery (see the
``dim_red.supcon.tails`` module docstring).

``train_classification_tail`` mirrors ``dim_red.autoencoder.training``'s
auxiliary-head loss shape (cross-entropy, family-masked spacegroup term).
``train_visualization_tail`` mirrors
``dim_red.supcon.training.train_supcon``'s loss shape (``supcon_loss``/
``norm_penalty``, same ``"random"``/``"balanced"`` batching choice)
computed on the visualization tail's own low-dimensional output rather than
a projection tail's.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from learned_optimization.research.general_lopt import prefab

from dim_red.supcon.sampling import iter_balanced_batches
from dim_red.supcon.tails import (
    ClassificationTail,
    VisualizationTail,
    apply_family_mask,
)
from dim_red.supcon.training import norm_penalty, supcon_loss

Array = jax.Array

logger = logging.getLogger("dim_red.pipeline")

_BATCHING_STRATEGIES = ("random", "balanced")
_DISTANCE_METRICS = ("euclidean", "cosine")
_OPTIMIZERS = ("adam", "velo")


@dataclass(frozen=True)
class TailTrainConfig:
    """Training configuration shared by ``train_classification_tail``/
    ``train_visualization_tail`` -- same shape/semantics as
    ``dim_red.supcon.training.TrainConfig``. ``tau``/``distance`` are only
    meaningful for ``train_visualization_tail``'s SupCon loss;
    ``train_classification_tail`` ignores them, same treatment other
    per-model-kind-irrelevant fields get elsewhere in this codebase.

    Attributes:
        epochs: Number of full passes over the training representations.
        batch_size: Number of rows per mini-batch.
        learning_rate: Adam's learning rate, used whenever ``optimizer ==
            "adam"`` (the default). Ignored (kept for API compatibility) when
            ``optimizer == "velo"``.
        optimizer: ``"adam"`` (default) -- a plain ``optax.adam(learning_rate)``
            -- or ``"velo"`` -- ``learned_optimization``'s pretrained VeLO
            meta-learned optimizer, whose ``num_steps``-dependent setup and
            pretrained-hypernetwork checkpoint load cost real, fixed time
            (several seconds or more) before training even starts -- a much
            bigger relative cost here than in phase 1, since these tails are
            tiny single-hidden-layer MLPs. See ``_make_optimizer``.
        tau: Temperature dividing similarities before the softmax inside
            ``train_visualization_tail``'s SupCon loss.
        distance: Which similarity ``train_visualization_tail``'s SupCon
            loss computes -- ``"euclidean"`` (default) or ``"cosine"``.
        seed: Seed controlling batch shuffling.
        device: JAX backend string (for example ``"cpu"`` or ``"gpu"``).
        early_stopping: If True, stop training once ``val_loss`` hasn't
            improved by more than ``early_stopping_min_delta`` for
            ``early_stopping_patience`` consecutive epochs.
        early_stopping_patience: Consecutive non-improving epochs tolerated
            before stopping. Ignored unless ``early_stopping`` is True.
        early_stopping_min_delta: Minimum decrease in ``val_loss`` counted
            as an improvement. Ignored unless ``early_stopping`` is True.
        early_stopping_restore_best: If True (default), the returned tail's
            params are the best-``val_loss`` epoch's rather than
            necessarily the last epoch trained. Ignored unless
            ``early_stopping`` is True.
    """

    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    tau: float = 0.1
    distance: str = "euclidean"
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


def _weighted_mean(values: List[float], weights: List[int]) -> float:
    values_arr = np.asarray(values, dtype=np.float64)
    weights_arr = np.asarray(weights, dtype=np.float64)
    return float(np.sum(values_arr * weights_arr) / np.sum(weights_arr))


def _validate_common(config: TailTrainConfig) -> None:
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
    if config.early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be a positive integer")
    if config.early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta must be >= 0")


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


def _device_for(config: TailTrainConfig):
    devices = jax.devices(config.device)
    if not devices:
        raise ValueError(f"No JAX devices found for backend '{config.device}'")
    return devices[0]


# --- Classification tail (cross-entropy) ------------------------------------


def _make_classification_train_step(
    tail: ClassificationTail,
    tx,
    lambda_family,
    lambda_spacegroup,
    has_family: bool,
    has_spacegroup: bool,
    family_spacegroup_mask,
):
    """Create a jitted training step bound to the classification tail,
    optimizer and loss weights. Only the tail's own params ever appear in
    ``params`` -- the frozen body is never touched here at all.
    """

    @jax.jit
    def _train_step(params, batch_r, batch_family, batch_spacegroup, opt_state):
        def loss_fn(local_params):
            total = jnp.asarray(0.0)
            family_ce = jnp.asarray(0.0)
            spacegroup_ce = jnp.asarray(0.0)
            if has_family:
                family_logits = tail.classify_family_with_params(local_params, batch_r)
                family_ce = optax.softmax_cross_entropy_with_integer_labels(
                    family_logits, batch_family
                ).mean()
                total = total + lambda_family * family_ce
            if has_spacegroup:
                spacegroup_logits = tail.classify_spacegroup_with_params(
                    local_params, batch_r
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


def _make_classification_eval_step(
    tail: ClassificationTail,
    lambda_family,
    lambda_spacegroup,
    has_family: bool,
    has_spacegroup: bool,
    family_spacegroup_mask,
):
    """Create a jitted evaluation step for one batch of representations."""

    @jax.jit
    def _eval_step(params, batch_r, batch_family, batch_spacegroup):
        total = jnp.asarray(0.0)
        family_ce = jnp.asarray(0.0)
        spacegroup_ce = jnp.asarray(0.0)
        if has_family:
            family_logits = tail.classify_family_with_params(params, batch_r)
            family_ce = optax.softmax_cross_entropy_with_integer_labels(
                family_logits, batch_family
            ).mean()
            total = total + lambda_family * family_ce
        if has_spacegroup:
            spacegroup_logits = tail.classify_spacegroup_with_params(params, batch_r)
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
        return total, family_ce, spacegroup_ce

    return _eval_step


def train_classification_tail(
    tail: ClassificationTail,
    r_train: np.ndarray,
    r_val: np.ndarray,
    config: TailTrainConfig,
    train_family_ids: Optional[np.ndarray] = None,
    val_family_ids: Optional[np.ndarray] = None,
    train_spacegroup_ids: Optional[np.ndarray] = None,
    val_spacegroup_ids: Optional[np.ndarray] = None,
    lambda_family: float = 1.0,
    lambda_spacegroup: float = 1.0,
    family_spacegroup_mask: Optional[np.ndarray] = None,
) -> Dict[str, List[float]]:
    """Train a ``ClassificationTail`` (Adam by default, or VeLO -- see
    ``TailTrainConfig.optimizer``) on a frozen SupCon body's precomputed
    representations, and return per-epoch loss history.

    Args:
        tail: ``ClassificationTail`` instance containing the Flax module and
            mutable parameters.
        r_train: Frozen body representations for the training split, shape
            ``(n_train, input_dim)``.
        r_val: Frozen body representations for the validation split.
        config: Training hyperparameters and execution backend options.
        train_family_ids: Integer family class ids (shape ``(n_train,)``),
            aligned row-for-row with ``r_train``. Together with
            ``val_family_ids`` and a tail built with a family head,
            activates the family cross-entropy term (weighted by
            ``lambda_family``).
        val_family_ids: Integer family class ids aligned with ``r_val``.
        train_spacegroup_ids: Integer spacegroup class ids (shape
            ``(n_train,)``), aligned with ``r_train``. Requires
            ``train_family_ids``/``val_family_ids`` and
            ``family_spacegroup_mask`` to also be given, and a tail built
            with a spacegroup head. Activates the family-masked spacegroup
            cross-entropy term (weighted by ``lambda_spacegroup``).
        val_spacegroup_ids: Integer spacegroup class ids aligned with ``r_val``.
        lambda_family: Weight of the family cross-entropy term.
        lambda_spacegroup: Weight of the (family-masked) spacegroup
            cross-entropy term.
        family_spacegroup_mask: ``1.0``/``0.0`` co-occurrence matrix of shape
            ``(n_family_classes, n_spacegroup_classes)``. See
            ``dim_red.supcon.tails.apply_family_mask``.

    Returns:
        Dictionary with per-epoch losses, one entry per epoch actually run
        (shorter than ``config.epochs`` if ``config.early_stopping`` stopped
        training early). Always contains ``"train_loss"``/``"val_loss"``
        (the full weighted objective). Also contains
        ``"train_family_ce"``/``"val_family_ce"`` when family ids are
        given, and ``"train_spacegroup_ce"``/``"val_spacegroup_ce"`` when
        spacegroup ids are given.

    Raises:
        ValueError: If config values are invalid, no matching JAX device is
            found, neither family nor spacegroup ids are given, or the
            family/spacegroup id arguments are inconsistent.
    """
    _validate_common(config)
    if lambda_family < 0:
        raise ValueError("lambda_family must be >= 0")
    if lambda_spacegroup < 0:
        raise ValueError("lambda_spacegroup must be >= 0")

    has_family = train_family_ids is not None
    has_spacegroup = train_spacegroup_ids is not None
    if not has_family and not has_spacegroup:
        raise ValueError(
            "At least one of train_family_ids/train_spacegroup_ids must be "
            "given -- there is nothing to classify otherwise"
        )
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

    device = _device_for(config)

    r_train_np = np.asarray(r_train, dtype=np.float32)
    r_val_np = np.asarray(r_val, dtype=np.float32)

    train_family_np = (
        np.asarray(train_family_ids, dtype=np.int32)
        if has_family
        else np.zeros(r_train_np.shape[0], dtype=np.int32)
    )
    val_family_np = (
        np.asarray(val_family_ids, dtype=np.int32)
        if has_family
        else np.zeros(r_val_np.shape[0], dtype=np.int32)
    )
    train_spacegroup_np = (
        np.asarray(train_spacegroup_ids, dtype=np.int32)
        if has_spacegroup
        else np.zeros(r_train_np.shape[0], dtype=np.int32)
    )
    val_spacegroup_np = (
        np.asarray(val_spacegroup_ids, dtype=np.int32)
        if has_spacegroup
        else np.zeros(r_val_np.shape[0], dtype=np.int32)
    )
    mask_jax = (
        jnp.asarray(family_spacegroup_mask, dtype=jnp.float32)
        if has_spacegroup
        else jnp.zeros((1, 1), dtype=jnp.float32)
    )

    rng = np.random.default_rng(config.seed)
    batches_per_epoch = int(np.ceil(r_train_np.shape[0] / config.batch_size))
    total_steps = max(1, config.epochs * batches_per_epoch)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    if has_family:
        history["train_family_ce"] = []
        history["val_family_ce"] = []
    if has_spacegroup:
        history["train_spacegroup_ce"] = []
        history["val_spacegroup_ce"] = []

    tx = _make_optimizer(config.optimizer, config.learning_rate, total_steps)
    lambda_family_jax = jnp.asarray(lambda_family, dtype=jnp.float32)
    lambda_spacegroup_jax = jnp.asarray(lambda_spacegroup, dtype=jnp.float32)

    train_step = _make_classification_train_step(
        tail,
        tx,
        lambda_family_jax,
        lambda_spacegroup_jax,
        has_family,
        has_spacegroup,
        mask_jax,
    )
    eval_step = _make_classification_eval_step(
        tail,
        lambda_family_jax,
        lambda_spacegroup_jax,
        has_family,
        has_spacegroup,
        mask_jax,
    )
    opt_state = tx.init(tail.params)
    params = tail.params

    best_val_loss = float("inf")
    best_params = None
    epochs_without_improvement = 0

    for _ in range(config.epochs):
        train_losses, train_ns = [], []
        train_family_ces, train_spacegroup_ces = [], []
        for batch_r, batch_family, batch_spacegroup in _iter_batches(
            (r_train_np, train_family_np, train_spacegroup_np), config.batch_size, rng
        ):
            batch_r_jax = jax.device_put(jnp.asarray(batch_r), device)
            batch_family_jax = jax.device_put(jnp.asarray(batch_family), device)
            batch_spacegroup_jax = jax.device_put(jnp.asarray(batch_spacegroup), device)
            params, opt_state, loss, family_ce, spacegroup_ce = train_step(
                params, batch_r_jax, batch_family_jax, batch_spacegroup_jax, opt_state
            )
            train_losses.append(float(loss))
            train_ns.append(batch_r.shape[0])
            if has_family:
                train_family_ces.append(float(family_ce))
            if has_spacegroup:
                train_spacegroup_ces.append(float(spacegroup_ce))

        val_losses, val_ns = [], []
        val_family_ces, val_spacegroup_ces = [], []
        for batch_r, batch_family, batch_spacegroup in _iter_batches(
            (r_val_np, val_family_np, val_spacegroup_np), config.batch_size, rng
        ):
            batch_r_jax = jax.device_put(jnp.asarray(batch_r), device)
            batch_family_jax = jax.device_put(jnp.asarray(batch_family), device)
            batch_spacegroup_jax = jax.device_put(jnp.asarray(batch_spacegroup), device)
            loss, family_ce, spacegroup_ce = eval_step(
                params, batch_r_jax, batch_family_jax, batch_spacegroup_jax
            )
            val_losses.append(float(loss))
            val_ns.append(batch_r.shape[0])
            if has_family:
                val_family_ces.append(float(family_ce))
            if has_spacegroup:
                val_spacegroup_ces.append(float(spacegroup_ce))

        history["train_loss"].append(_weighted_mean(train_losses, train_ns))
        history["val_loss"].append(_weighted_mean(val_losses, val_ns))
        if has_family:
            history["train_family_ce"].append(
                _weighted_mean(train_family_ces, train_ns)
            )
            history["val_family_ce"].append(_weighted_mean(val_family_ces, val_ns))
        if has_spacegroup:
            history["train_spacegroup_ce"].append(
                _weighted_mean(train_spacegroup_ces, train_ns)
            )
            history["val_spacegroup_ce"].append(
                _weighted_mean(val_spacegroup_ces, val_ns)
            )

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
    tail.params = params
    return history


# --- Visualization tail (SupCon loss) ---------------------------------------


def _make_visualization_train_step(
    tail: VisualizationTail,
    tx,
    tau: float,
    distance: str,
    lambda_family,
    lambda_spacegroup,
    lambda_norm,
    has_family: bool,
    has_spacegroup: bool,
):
    """Create a jitted training step bound to the visualization tail,
    optimizer and loss weights. Only the tail's own params ever appear in
    ``params`` -- the frozen body is never touched here at all.
    """

    @jax.jit
    def _train_step(params, batch_r, batch_family, batch_spacegroup, opt_state):
        def loss_fn(local_params):
            z = tail.project_with_params(local_params, batch_r)
            total = jnp.asarray(0.0)
            family_loss = jnp.asarray(0.0)
            spacegroup_loss = jnp.asarray(0.0)
            if has_family:
                family_loss = supcon_loss(z, batch_family, tau, distance)
                total = total + lambda_family * family_loss
            if has_spacegroup:
                spacegroup_loss = supcon_loss(z, batch_spacegroup, tau, distance)
                total = total + lambda_spacegroup * spacegroup_loss
            penalty = norm_penalty(z)
            total = total + lambda_norm * penalty
            return total, (family_loss, spacegroup_loss, penalty)

        (loss, (family_loss, spacegroup_loss, penalty)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)
        updates, new_opt_state = tx.update(
            grads, opt_state, params, extra_args={"loss": loss}
        )
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, family_loss, spacegroup_loss, penalty

    return _train_step


def _make_visualization_eval_step(
    tail: VisualizationTail,
    tau: float,
    distance: str,
    lambda_family,
    lambda_spacegroup,
    lambda_norm,
    has_family: bool,
    has_spacegroup: bool,
):
    """Create a jitted evaluation step for one (unpadded) batch of representations."""

    @jax.jit
    def _eval_step(params, batch_r, batch_family, batch_spacegroup):
        z = tail.project_with_params(params, batch_r)
        total = jnp.asarray(0.0)
        family_loss = jnp.asarray(0.0)
        spacegroup_loss = jnp.asarray(0.0)
        if has_family:
            family_loss = supcon_loss(z, batch_family, tau, distance)
            total = total + lambda_family * family_loss
        if has_spacegroup:
            spacegroup_loss = supcon_loss(z, batch_spacegroup, tau, distance)
            total = total + lambda_spacegroup * spacegroup_loss
        penalty = norm_penalty(z)
        total = total + lambda_norm * penalty
        return total, family_loss, spacegroup_loss, penalty

    return _eval_step


def train_visualization_tail(
    tail: VisualizationTail,
    r_train: np.ndarray,
    r_val: np.ndarray,
    config: TailTrainConfig,
    train_family_ids: Optional[np.ndarray] = None,
    val_family_ids: Optional[np.ndarray] = None,
    train_spacegroup_ids: Optional[np.ndarray] = None,
    val_spacegroup_ids: Optional[np.ndarray] = None,
    lambda_family: float = 1.0,
    lambda_spacegroup: float = 1.0,
    lambda_norm: float = 0.0,
    batching_strategy: str = "random",
    batching_family_ids: Optional[np.ndarray] = None,
    batching_spacegroup_ids: Optional[np.ndarray] = None,
    batching_P: Optional[int] = None,
    batching_K: Optional[int] = None,
    batching_S: Optional[int] = None,
) -> Dict[str, List[float]]:
    """Train a ``VisualizationTail`` (Adam by default, or VeLO -- see
    ``TailTrainConfig.optimizer``) on a frozen SupCon body's precomputed
    representations, using the same Supervised Contrastive loss
    as phase 1's projection tail, and return per-epoch loss history.

    Mirrors ``dim_red.supcon.training.train_supcon``'s arguments/behavior
    (same batching choice, same history keys) -- the only difference is
    that this trains a ``VisualizationTail`` directly on precomputed
    representations, with no body forward pass at all.

    Args:
        tail: ``VisualizationTail`` instance containing the Flax module and
            mutable parameters.
        r_train: Frozen body representations for the training split, shape
            ``(n_train, input_dim)``.
        r_val: Frozen body representations for the validation split.
        config: Training hyperparameters and execution backend options.
        train_family_ids: Integer family class ids aligned with ``r_train``.
            Together with ``val_family_ids``, activates the family-level
            SupCon term.
        val_family_ids: Integer family class ids aligned with ``r_val``.
        train_spacegroup_ids: Integer spacegroup class ids aligned with
            ``r_train``. Together with ``val_spacegroup_ids``, activates
            the spacegroup-level SupCon term.
        val_spacegroup_ids: Integer spacegroup class ids aligned with ``r_val``.
        lambda_family: Weight of the family-level term.
        lambda_spacegroup: Weight of the spacegroup-level term.
        lambda_norm: Weight of the embedding-norm regularizer (see
            ``dim_red.supcon.training.norm_penalty``), applied to this
            tail's own low-dimensional output.
        batching_strategy: ``"random"`` (default) or ``"balanced"`` -- see
            ``dim_red.supcon.training.train_supcon``. Only affects training
            batches; validation always stays ``"random"``.
        batching_family_ids: Integer family id per training row, required
            when ``batching_strategy == "balanced"``.
        batching_spacegroup_ids: Integer spacegroup id per training row,
            required when ``batching_strategy == "balanced"``.
        batching_P: Number of families per balanced batch.
        batching_K: Number of examples per family per balanced batch.
            Required (positive integer) when ``batching_strategy ==
            "balanced"``.
        batching_S: Number of spacegroups to stratify by within each chosen
            family.

    Returns:
        Dictionary with per-epoch losses -- same shape as
        ``dim_red.supcon.training.train_supcon``'s returned history.

    Raises:
        ValueError: If config values are invalid, no matching JAX device is
            found, neither family nor spacegroup ids are given, the
            family/spacegroup id arguments are inconsistent, or
            ``batching_strategy`` is invalid or missing its required
            ``batching_*`` arguments.
    """
    _validate_common(config)
    if config.tau <= 0:
        raise ValueError("tau must be > 0")
    if config.distance not in _DISTANCE_METRICS:
        raise ValueError(
            f"distance must be one of {_DISTANCE_METRICS}, got {config.distance!r}"
        )
    if lambda_family < 0:
        raise ValueError("lambda_family must be >= 0")
    if lambda_spacegroup < 0:
        raise ValueError("lambda_spacegroup must be >= 0")
    if lambda_norm < 0:
        raise ValueError("lambda_norm must be >= 0")
    if batching_strategy not in _BATCHING_STRATEGIES:
        raise ValueError(
            f"batching_strategy must be one of {_BATCHING_STRATEGIES}, got "
            f"{batching_strategy!r}"
        )

    has_family = train_family_ids is not None
    has_spacegroup = train_spacegroup_ids is not None
    if not has_family and not has_spacegroup:
        raise ValueError(
            "At least one of train_family_ids/train_spacegroup_ids must be "
            "given -- there is nothing to contrast on otherwise"
        )
    if has_family != (val_family_ids is not None):
        raise ValueError("train_family_ids and val_family_ids must be given together")
    if has_spacegroup != (val_spacegroup_ids is not None):
        raise ValueError(
            "train_spacegroup_ids and val_spacegroup_ids must be given together"
        )

    balanced_batching = batching_strategy == "balanced"
    if balanced_batching:
        if batching_family_ids is None:
            raise ValueError(
                "batching_strategy='balanced' requires batching_family_ids "
                "(balanced batches are always grouped by family first, "
                "independent of whether the family SupCon loss term itself "
                "is active)"
            )
        if batching_K is None or batching_K <= 0:
            raise ValueError(
                "batching_K must be a positive integer when "
                "batching_strategy='balanced'"
            )
        if batching_spacegroup_ids is None:
            raise ValueError(
                "batching_strategy='balanced' requires batching_spacegroup_ids "
                "(spacegroup stratification is always applied -- batching_S="
                "None means 'use every spacegroup present per family', not "
                "'skip spacegroup stratification')"
            )

    device = _device_for(config)

    r_train_np = np.asarray(r_train, dtype=np.float32)
    r_val_np = np.asarray(r_val, dtype=np.float32)

    train_family_np = (
        np.asarray(train_family_ids, dtype=np.int32)
        if has_family
        else np.zeros(r_train_np.shape[0], dtype=np.int32)
    )
    val_family_np = (
        np.asarray(val_family_ids, dtype=np.int32)
        if has_family
        else np.zeros(r_val_np.shape[0], dtype=np.int32)
    )
    train_spacegroup_np = (
        np.asarray(train_spacegroup_ids, dtype=np.int32)
        if has_spacegroup
        else np.zeros(r_train_np.shape[0], dtype=np.int32)
    )
    val_spacegroup_np = (
        np.asarray(val_spacegroup_ids, dtype=np.int32)
        if has_spacegroup
        else np.zeros(r_val_np.shape[0], dtype=np.int32)
    )

    if balanced_batching:
        batching_family_np = np.asarray(batching_family_ids, dtype=np.int32)
        batching_spacegroup_np = np.asarray(batching_spacegroup_ids, dtype=np.int32)
        n_train_families = len(np.unique(batching_family_np))
        effective_P = (
            n_train_families
            if batching_P is None
            else min(batching_P, n_train_families)
        )
        effective_batch_size = effective_P * batching_K
        logger.warning(
            "batching_strategy='balanced': batch_size (%d) is ignored for "
            "training batches -- effective batch size = P*K = %d*%d = %d. "
            "Validation batches are unaffected (still random shuffle, using "
            "batch_size=%d).",
            config.batch_size,
            effective_P,
            batching_K,
            effective_batch_size,
            config.batch_size,
        )
    else:
        effective_batch_size = config.batch_size

    rng = np.random.default_rng(config.seed)
    batches_per_epoch = int(np.ceil(r_train_np.shape[0] / effective_batch_size))
    total_steps = max(1, config.epochs * batches_per_epoch)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_norm_penalty": [],
        "val_norm_penalty": [],
    }
    if has_family:
        history["train_family_supcon"] = []
        history["val_family_supcon"] = []
    if has_spacegroup:
        history["train_spacegroup_supcon"] = []
        history["val_spacegroup_supcon"] = []

    tx = _make_optimizer(config.optimizer, config.learning_rate, total_steps)
    lambda_family_jax = jnp.asarray(lambda_family, dtype=jnp.float32)
    lambda_spacegroup_jax = jnp.asarray(lambda_spacegroup, dtype=jnp.float32)
    lambda_norm_jax = jnp.asarray(lambda_norm, dtype=jnp.float32)

    train_step = _make_visualization_train_step(
        tail,
        tx,
        config.tau,
        config.distance,
        lambda_family_jax,
        lambda_spacegroup_jax,
        lambda_norm_jax,
        has_family,
        has_spacegroup,
    )
    eval_step = _make_visualization_eval_step(
        tail,
        config.tau,
        config.distance,
        lambda_family_jax,
        lambda_spacegroup_jax,
        lambda_norm_jax,
        has_family,
        has_spacegroup,
    )
    opt_state = tx.init(tail.params)
    params = tail.params

    best_val_loss = float("inf")
    best_params = None
    epochs_without_improvement = 0

    for _ in range(config.epochs):
        (
            train_losses,
            train_family_losses,
            train_spacegroup_losses,
            train_norm_penalties,
            train_ns,
        ) = ([], [], [], [], [])
        if balanced_batching:
            train_batches = iter_balanced_batches(
                (r_train_np, train_family_np, train_spacegroup_np),
                batching_family_np,
                batching_spacegroup_np,
                batching_P,
                batching_K,
                batching_S,
                batches_per_epoch,
                rng,
            )
        else:
            train_batches = _iter_batches(
                (r_train_np, train_family_np, train_spacegroup_np),
                effective_batch_size,
                rng,
            )
        for batch_r, batch_family, batch_spacegroup in train_batches:
            batch_r_jax = jax.device_put(jnp.asarray(batch_r), device)
            batch_family_jax = jax.device_put(jnp.asarray(batch_family), device)
            batch_spacegroup_jax = jax.device_put(jnp.asarray(batch_spacegroup), device)
            params, opt_state, loss, family_loss, spacegroup_loss, penalty = train_step(
                params, batch_r_jax, batch_family_jax, batch_spacegroup_jax, opt_state
            )
            train_losses.append(float(loss))
            train_ns.append(batch_r.shape[0])
            train_norm_penalties.append(float(penalty))
            if has_family:
                train_family_losses.append(float(family_loss))
            if has_spacegroup:
                train_spacegroup_losses.append(float(spacegroup_loss))

        (
            val_losses,
            val_family_losses,
            val_spacegroup_losses,
            val_norm_penalties,
            val_ns,
        ) = ([], [], [], [], [])
        for batch_r, batch_family, batch_spacegroup in _iter_batches(
            (r_val_np, val_family_np, val_spacegroup_np), config.batch_size, rng
        ):
            batch_r_jax = jax.device_put(jnp.asarray(batch_r), device)
            batch_family_jax = jax.device_put(jnp.asarray(batch_family), device)
            batch_spacegroup_jax = jax.device_put(jnp.asarray(batch_spacegroup), device)
            loss, family_loss, spacegroup_loss, penalty = eval_step(
                params, batch_r_jax, batch_family_jax, batch_spacegroup_jax
            )
            val_losses.append(float(loss))
            val_ns.append(batch_r.shape[0])
            val_norm_penalties.append(float(penalty))
            if has_family:
                val_family_losses.append(float(family_loss))
            if has_spacegroup:
                val_spacegroup_losses.append(float(spacegroup_loss))

        history["train_loss"].append(_weighted_mean(train_losses, train_ns))
        history["val_loss"].append(_weighted_mean(val_losses, val_ns))
        history["train_norm_penalty"].append(
            _weighted_mean(train_norm_penalties, train_ns)
        )
        history["val_norm_penalty"].append(_weighted_mean(val_norm_penalties, val_ns))
        if has_family:
            history["train_family_supcon"].append(
                _weighted_mean(train_family_losses, train_ns)
            )
            history["val_family_supcon"].append(
                _weighted_mean(val_family_losses, val_ns)
            )
        if has_spacegroup:
            history["train_spacegroup_supcon"].append(
                _weighted_mean(train_spacegroup_losses, train_ns)
            )
            history["val_spacegroup_supcon"].append(
                _weighted_mean(val_spacegroup_losses, val_ns)
            )

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
    tail.params = params
    return history
