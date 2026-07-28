"""
Training utilities for VAE models.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from learned_optimization.research.general_lopt import prefab

from dim_red.vae.database import VAEDatabase
from dim_red.vae.model import VAE

Array = jax.Array


@dataclass(frozen=True)
class TrainConfig:
    """Training configuration for VAE optimization.

    Attributes:
        epochs: Number of full passes over training data.
        batch_size: Number of samples per mini-batch.
        learning_rate: Kept for API compatibility; VeLO is still used as
            optimizer backend.
        beta: Weight applied to the KL-divergence term in VAE loss.
        seed: Seed controlling batch shuffling and random latent sampling.
        device: JAX backend string (for example ``"cpu"`` or ``"gpu"``).
    """

    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    beta: float = 1.0
    seed: int = 42
    device: str = "cpu"


def vae_loss(
    x_recon: Array, x_true: Array, mu: Array, logvar: Array, beta: float = 1.0
) -> tuple[Array, Array, Array]:
    """Compute the beta-VAE loss components.

    Args:
        x_recon: Reconstructed batch.
        x_true: Ground-truth batch.
        mu: Mean of latent posterior.
        logvar: Log-variance of latent posterior.
        beta: Multiplicative factor for KL regularization.

    Returns:
        Tuple ``(total, recon, kl)`` where:
        - ``total`` is the objective optimized during training,
        - ``recon`` is reconstruction MSE,
        - ``kl`` is KL divergence against a unit Gaussian prior.
    """
    recon = jnp.mean((x_recon - x_true) ** 2)
    kl = -0.5 * jnp.mean(1.0 + logvar - mu**2 - jnp.exp(logvar))
    total = recon + beta * kl
    return total, recon, kl


def _iter_batches(
    X: np.ndarray, batch_size: int, rng: np.random.Generator
) -> List[np.ndarray]:
    """Shuffle and split data into mini-batches."""
    n_samples = X.shape[0]
    indices = rng.permutation(n_samples)
    batches = []
    for start in range(0, n_samples, batch_size):
        batch_idx = indices[start : start + batch_size]
        batches.append(X[batch_idx])
    return batches


def _make_train_step(model: VAE, tx, beta):
    """Create a jitted training step bound to model, optimizer and beta."""

    @jax.jit
    def _train_step(params, batch_x, key, opt_state):
        """Single optimization step returning updated params/state/losses."""

        def loss_fn(local_params):
            mu, logvar = model.encode_with_params(local_params, batch_x)
            z = model.reparameterize(key, mu, logvar)
            x_recon = model.decode_with_params(local_params, z)
            total, recon, kl = vae_loss(x_recon, batch_x, mu, logvar, beta=beta)
            return total, (recon, kl)

        (loss, (recon, kl)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = tx.update(
            grads,
            opt_state,
            params,
            extra_args={"loss": loss},
        )
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, recon, kl

    return _train_step


def _make_eval_step(model: VAE, beta):
    """Create a jitted evaluation step returning per-sample losses."""

    @jax.jit
    def _eval_step(params, batch_x, key):
        """Compute per-sample validation total/recon/KL losses for one batch."""
        mu, logvar = model.encode_with_params(params, batch_x)
        z = model.reparameterize(key, mu, logvar)
        x_recon = model.decode_with_params(params, z)
        recon = jnp.mean((x_recon - batch_x) ** 2, axis=1)
        kl = -0.5 * jnp.mean(1.0 + logvar - mu**2 - jnp.exp(logvar), axis=1)
        total = recon + beta * kl
        return total, recon, kl

    return _eval_step


def _prepare_padded_batches(
    X: np.ndarray, batch_size: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Create fixed-size shuffled batches and a validity mask for padding.

    Returns:
        A tuple ``(batched_x, batched_mask)`` where:
        - ``batched_x`` has shape ``(n_batches, batch_size, n_features)``.
        - ``batched_mask`` has shape ``(n_batches, batch_size)`` with 1.0 on
          valid samples and 0.0 on padded entries.
    """
    n_samples, n_features = X.shape
    indices = rng.permutation(n_samples)
    X_shuffled = X[indices]
    n_batches = int(np.ceil(n_samples / batch_size))
    total_slots = n_batches * batch_size
    pad = total_slots - n_samples

    if pad > 0:
        X_shuffled = np.pad(X_shuffled, ((0, pad), (0, 0)), mode="constant")

    batched_x = X_shuffled.reshape(n_batches, batch_size, n_features)
    batched_mask = np.ones((n_batches, batch_size), dtype=np.float32)
    if pad > 0:
        batched_mask[-1, batch_size - pad :] = 0.0
    return batched_x, batched_mask


def _make_eval_epoch(eval_step):
    """Create a jitted epoch-level parallel evaluator over mini-batches."""

    @jax.jit
    def _eval_epoch(params, batched_x, batched_mask, keys):
        total, recon, kl = jax.vmap(eval_step, in_axes=(None, 0, 0))(
            params, batched_x, keys
        )
        valid_count = jnp.maximum(jnp.sum(batched_mask), 1.0)
        mean_total = jnp.sum(total * batched_mask) / valid_count
        mean_recon = jnp.sum(recon * batched_mask) / valid_count
        mean_kl = jnp.sum(kl * batched_mask) / valid_count
        return mean_total, mean_recon, mean_kl

    return _eval_epoch


def train_vae(
    model: VAE, train_db: VAEDatabase, val_db: VAEDatabase, config: TrainConfig
) -> Dict[str, List[float]]:
    """Train a VAE with VeLO (Optax wrapper) and return loss history.

    Args:
        model: VAE instance containing Flax module and mutable parameters.
        train_db: Training dataset wrapper.
        val_db: Validation dataset wrapper.
        config: Training hyperparameters and execution backend options.

    Returns:
        Dictionary containing per-epoch losses with keys ``"train_loss"``,
        ``"train_recon"``, ``"train_kl"``, ``"val_loss"``, ``"val_recon"``
        and ``"val_kl"``. The ``*_loss`` entries are the beta-weighted total
        objective; ``*_recon`` and ``*_kl`` are its two components,
        unweighted by ``beta``.

    Raises:
        ValueError: If config values are invalid or requested JAX backend
            has no available devices.
    """
    if config.epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")
    if config.beta < 0:
        raise ValueError("beta must be >= 0")

    devices = jax.devices(config.device)
    if not devices:
        raise ValueError(f"No JAX devices found for backend '{config.device}'")
    device = devices[0]

    train_np = np.asarray(train_db.data, dtype=np.float32)
    val_np = np.asarray(val_db.data, dtype=np.float32)
    rng = np.random.default_rng(config.seed)
    jax_key = jax.random.PRNGKey(config.seed)
    batches_per_epoch = int(np.ceil(train_np.shape[0] / config.batch_size))
    total_steps = max(1, config.epochs * batches_per_epoch)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_recon": [],
        "train_kl": [],
        "val_loss": [],
        "val_recon": [],
        "val_kl": [],
    }
    tx = prefab.optax_lopt(num_steps=total_steps)
    beta = jnp.asarray(config.beta, dtype=jnp.float32)
    train_step = _make_train_step(model, tx, beta)
    eval_step = _make_eval_step(model, beta)
    eval_epoch = _make_eval_epoch(eval_step)
    opt_state = tx.init(model.params)
    params = model.params

    for _ in range(config.epochs):
        train_losses = []
        train_recons = []
        train_kls = []
        train_batches = _iter_batches(train_np, config.batch_size, rng)
        for batch in train_batches:
            jax_key, step_key = jax.random.split(jax_key)
            batch_x = jax.device_put(jnp.asarray(batch), device)
            params, opt_state, loss, recon, kl = train_step(
                params, batch_x, step_key, opt_state
            )
            train_losses.append(float(loss))
            train_recons.append(float(recon))
            train_kls.append(float(kl))

        val_batched_x, val_batched_mask = _prepare_padded_batches(
            val_np, config.batch_size, rng
        )
        val_batched_x_jax = jax.device_put(jnp.asarray(val_batched_x), device)
        val_batched_mask_jax = jax.device_put(jnp.asarray(val_batched_mask), device)
        n_val_batches = val_batched_x.shape[0]
        split_keys = jax.random.split(jax_key, n_val_batches + 1)
        jax_key = split_keys[0]
        val_keys = split_keys[1:]
        val_loss, val_recon, val_kl = eval_epoch(
            params, val_batched_x_jax, val_batched_mask_jax, val_keys
        )

        history["train_loss"].append(float(np.mean(train_losses)))
        history["train_recon"].append(float(np.mean(train_recons)))
        history["train_kl"].append(float(np.mean(train_kls)))
        history["val_loss"].append(float(val_loss))
        history["val_recon"].append(float(val_recon))
        history["val_kl"].append(float(val_kl))

    model.params = params
    return history
