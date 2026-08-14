"""
Training utilities for the SupCon (Supervised Contrastive) body + projection
tail (phase 1 -- see ``dim_red.supcon.tail_training`` for phase 2, training a
classification/visualization tail on the frozen body afterwards).

Mirrors ``dim_red.autoencoder.training``'s overall shape (``TrainConfig``,
VeLO as the optimizer backend, the same ``has_family``/``has_spacegroup``
compile-time-branch pattern) but the objective is entirely different: no
reconstruction, no classifier heads -- a Supervised Contrastive loss
computed against family and/or spacegroup labels, on the *projection tail's*
output (Khosla et al. 2020's ``z = Proj(Enc(x))``), not directly on the
body's own representation ``r`` -- see ``train_supcon``.

Validation deliberately does **not** reuse the padding+``vmap``-over-epoch
trick used by ``vae.training``/``autoencoder.training``: that trick zero-pads
the last batch and relies on every per-sample loss term being independent of
the other rows in its batch (true for MSE/cross-entropy, computed per row).
SupCon is *pairwise* within a batch -- a zero-padded dummy row would become a
spurious positive/negative for real rows instead of being cleanly maskable
after the fact. Both training and validation therefore use a plain,
unpadded batch loop (one jitted call per batch), with epoch-level metrics
combined as a batch-size-weighted mean.

Two mutually-exclusive strategies build *training* batches (``train_supcon``'s
``batching_strategy`` argument): ``"random"`` (default, a plain shuffle via
``_iter_batches``, unchanged) or ``"balanced"`` (a family/spacegroup-
stratified sampler, see ``dim_red.supcon.sampling``, that guarantees
positives are available for the SupCon loss even on less-balanced datasets
than the pyxtal-generated one this package was built against). Validation
batches always stay ``"random"`` regardless of this setting, by design --
see ``train_supcon``'s docstring.

Optional early stopping (``TrainConfig.early_stopping``, disabled by
default) monitors ``val_loss`` and stops once it hasn't improved for
``early_stopping_patience`` consecutive epochs -- same mechanism (and same
``TrainConfig`` field names) as ``vae.training``/``autoencoder.training``.

An optional embedding-norm regularizer (``train_supcon``'s ``lambda_norm``
argument, ``0.0`` -- disabled -- by default) adds ``lambda_norm *
norm_penalty(z)`` to the total loss. Unlike ``lambda_family``/
``lambda_spacegroup``, it isn't gated by ``mode``: it doesn't depend on any
label, so it's simply in effect whenever it's non-zero. It exists because
``supcon_loss``'s default ``distance == "euclidean"`` similarity has no
built-in scale normalization, so nothing otherwise stops the encoder from
inflating ``z``'s norm without bound (with ``distance == "cosine"``,
similarity is already bounded to ``[-1, 1]`` by the L2-normalization, so
``lambda_norm`` is typically unnecessary there).
``lambda_norm`` follows the same "config-layer input, not part of
``TrainConfig``" placement as ``lambda_family``/``lambda_spacegroup`` (see
``dim_red.pipeline.config.SupConConfig.lambda_norm``) since it's a loss-term
weight, not a training-loop mechanic like ``tau``/``epochs``/``device``.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dim_red.supcon.model import SupConEncoder
from dim_red.supcon.sampling import iter_balanced_batches
from dim_red.supcon.tails import ProjectionTail
from dim_red.vae.database import VAEDatabase

Array = jax.Array

logger = logging.getLogger("dim_red.pipeline")

_BATCHING_STRATEGIES = ("random", "balanced")
_DISTANCE_METRICS = ("euclidean", "cosine")
_OPTIMIZERS = ("adam", "velo")


@dataclass(frozen=True)
class TrainConfig:
    """Training configuration for ``SupConEncoder`` optimization.

    Deliberately does *not* hold ``lambda_family``/``lambda_spacegroup``
    (unlike ``vae.training.TrainConfig``/``autoencoder.training.TrainConfig``):
    those live in ``dim_red.pipeline.config.SupConConfig`` instead and are
    passed to ``train_supcon`` as explicit arguments, since ``mode``
    (which label level(s) are active) lives there too and the two are best
    kept together at the config layer.

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
            (several seconds or more) before training even starts. See
            ``_make_optimizer``.
        tau: Temperature dividing similarities before the softmax inside the
            SupCon loss.
        distance: Which similarity ``supcon_loss`` computes: ``"euclidean"``
            (default, current/original behavior -- negative squared
            Euclidean distance, unbounded scale, pairs with
            ``lambda_norm``/``norm_penalty`` if that's a problem) or
            ``"cosine"`` (cosine similarity between L2-normalized ``z``
            vectors -- bounded to ``[-1, 1]`` by construction, so
            ``lambda_norm`` is typically unnecessary with it).
        seed: Seed controlling batch shuffling.
        device: JAX backend string (for example ``"cpu"`` or ``"gpu"``).
        early_stopping: If True, stop training once ``val_loss`` (the full
            weighted objective) hasn't improved by more than
            ``early_stopping_min_delta`` for ``early_stopping_patience``
            consecutive epochs. Disabled by default (current behavior
            unchanged) -- the monitored metric is always ``val_loss``, not
            configurable.
        early_stopping_patience: Consecutive non-improving epochs tolerated
            before stopping. Ignored unless ``early_stopping`` is True.
        early_stopping_min_delta: Minimum decrease in ``val_loss`` counted
            as an improvement. Ignored unless ``early_stopping`` is True.
        early_stopping_restore_best: If True (default), the returned
            ``model.params`` are the best-``val_loss`` epoch's rather than
            necessarily the last epoch trained -- whether training stopped
            early or ran the full ``epochs``. Ignored unless
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


def supcon_loss(
    z: Array,
    labels: Array,
    tau: float,
    distance: str = "euclidean",
    eps: float = 1e-8,
) -> Array:
    """Supervised Contrastive loss (Khosla et al. 2020) for one batch and one
    label level.

    For anchor ``i``, positives ``P(i)`` are other batch rows sharing
    ``labels[i]``; the denominator set ``A(i)`` is every other batch row
    (``i`` excluded). Anchors with an empty ``P(i)`` (no other row in this
    batch shares their label) contribute 0 to the sum *and* are excluded
    from the count the mean divides by -- i.e. excluded from that term's
    computation for this step, rather than folded in as a zero that would
    otherwise dilute the mean over the full batch size. A batch of size < 2
    (no possible positives or negatives at all) short-circuits to 0.0 before
    any similarity/normalization math runs -- this is a static Python branch
    (batch size is a shape, known at trace time), not a data-dependent one,
    so it also sidesteps the ``logsumexp`` of an all-``-inf`` row that would
    otherwise poison gradients in that degenerate case.

    Args:
        z: Latent batch, shape ``(batch, latent_dim)``. Only L2-normalized
            internally when ``distance == "cosine"``; for ``"euclidean"``
            (default) ``z``'s own scale matters, since similarity is negative
            squared Euclidean distance with no fixed range -- pair with
            ``lambda_norm``/``norm_penalty`` if that's a problem.
        labels: Integer label ids, shape ``(batch,)``.
        tau: Temperature dividing similarities before the softmax.
        distance: ``"euclidean"`` (default -- negative squared Euclidean
            distance, ``sim_ij = -||z_i - z_j||^2``, the original formulation
            here) or ``"cosine"`` (cosine similarity between L2-normalized
            vectors, ``sim_ij = (z_i / ||z_i||) . (z_j / ||z_j||)``, bounded
            to ``[-1, 1]`` -- closer to the more common SupCon formulation).
        eps: Added under the square root when normalizing ``z`` for
            ``distance == "cosine"`` to avoid dividing by zero for a
            near-zero-norm row. Unused for ``distance == "euclidean"``.

    Returns:
        Scalar loss, the mean over anchors with ``|P(i)| > 0`` of
        ``-log(sum_{p in P(i)} exp(sim_ip/tau) / sum_{a in A(i)} exp(sim_ia/tau))``
        (``0.0`` if no anchor in the batch has any positive), where ``sim``
        is chosen by ``distance`` (see above).

    Raises:
        ValueError: If ``distance`` is not one of ``"euclidean"``/``"cosine"``.
    """
    if distance not in _DISTANCE_METRICS:
        raise ValueError(
            f"distance must be one of {_DISTANCE_METRICS}, got {distance!r}"
        )

    batch_size = z.shape[0]
    if batch_size < 2:
        return jnp.asarray(0.0, dtype=z.dtype)

    if distance == "cosine":
        z_norm = z / jnp.sqrt(jnp.sum(z**2, axis=-1, keepdims=True) + eps)
        sim = (z_norm @ z_norm.T) / tau
    else:
        sq_norms = jnp.sum(z**2, axis=-1, keepdims=True)
        sq_dists = sq_norms + sq_norms.T - 2 * (z @ z.T)
        # Floating-point error can push same-point distances slightly negative.
        sq_dists = jnp.maximum(sq_dists, 0.0)
        sim = -sq_dists / tau

    self_mask = jnp.eye(batch_size, dtype=bool)
    positive_mask = (labels[:, None] == labels[None, :]) & ~self_mask
    positive_mask_f = positive_mask.astype(z.dtype)

    # A large *finite* negative sentinel, not -jnp.inf: log_softmax's output
    # at a masked-out (self) position is itself then a large-but-finite
    # negative number rather than a literal -inf. That matters because that
    # position always gets multiplied by 0 (self is never a positive) right
    # below -- 0 * (-inf) is NaN in IEEE754, silently poisoning the whole
    # row's sum on the *forward* pass (not just the gradient), whereas
    # 0 * (large finite number) is a clean 0.0.
    neg_sentinel = jnp.asarray(-1e9, dtype=z.dtype)
    sim_for_denom = jnp.where(self_mask, neg_sentinel, sim)
    log_prob = jax.nn.log_softmax(sim_for_denom, axis=-1)

    pos_count = positive_mask_f.sum(axis=-1)
    has_positive = pos_count > 0
    per_anchor_loss = -jnp.sum(positive_mask_f * log_prob, axis=-1) / jnp.maximum(
        pos_count, 1.0
    )
    per_anchor_loss = jnp.where(has_positive, per_anchor_loss, 0.0)

    n_valid = jnp.maximum(jnp.sum(has_positive.astype(z.dtype)), 1.0)
    return jnp.sum(per_anchor_loss) / n_valid


def norm_penalty(z: Array) -> Array:
    """Mean squared L2 norm of a batch's latent vectors -- an optional
    regularizer (weighted by ``lambda_norm``, see this module's docstring)
    discouraging unbounded embedding growth.

    Matters specifically for ``supcon_loss``'s default ``distance ==
    "euclidean"`` similarity, which has no built-in scale normalization:
    without this penalty, nothing stops the encoder from trivially shrinking
    pairwise distances between same-label points by inflating ``z``'s norm
    overall, rather than actually learning meaningful directions. Largely
    redundant with ``distance == "cosine"``, whose L2-normalization already
    makes similarity scale-invariant by construction.

    Args:
        z: Latent batch, shape ``(batch, latent_dim)``.

    Returns:
        Scalar: ``mean_i(||z_i||^2)``.
    """
    return jnp.mean(jnp.sum(z**2, axis=-1))


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
        from learned_optimization.research.general_lopt import prefab

        return prefab.optax_lopt(num_steps=num_steps)
    return optax.with_extra_args_support(optax.adam(learning_rate))


def _make_train_step(
    model: SupConEncoder,
    projection_tail: ProjectionTail,
    tx,
    tau: float,
    distance: str,
    lambda_family,
    lambda_spacegroup,
    lambda_norm,
    has_family: bool,
    has_spacegroup: bool,
):
    """Create a jitted training step bound to body, projection tail,
    optimizer and loss weights.

    ``params`` is a plain ``{"body": ..., "tail": ...}`` dict composed from
    the body's and projection tail's independently-initialized pytrees --
    both are differentiated together by the single ``jax.value_and_grad``
    call below, so this is where body+tail are jointly trained (see the
    module docstring). ``supcon_loss``/``norm_penalty`` are computed on the
    projection tail's output ``z = Proj(r)``, not the body's raw
    representation ``r``.

    ``has_family``/``has_spacegroup`` are plain Python bools (not traced
    values): the branches they guard are resolved at trace time, so the
    inactive term's ``supcon_loss`` call is compiled away entirely -- same
    pattern already used by ``vae.training``/``autoencoder.training`` for
    their auxiliary heads. ``norm_penalty`` (see that function) is always
    computed and returned regardless of ``lambda_norm`` -- like
    ``lambda_family``/``lambda_spacegroup``, a weight of ``0.0`` simply
    contributes nothing to ``total`` (compiled away at trace time via
    constant folding, no separate Python bool needed to gate it), while the
    unweighted value stays available for per-epoch logging either way.
    """

    @jax.jit
    def _train_step(params, batch_x, batch_family, batch_spacegroup, opt_state):
        """Single optimization step returning updated params/state/losses."""

        def loss_fn(local_params):
            r = model.encode_with_params(local_params["body"], batch_x)
            z = projection_tail.project_with_params(local_params["tail"], r)
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


def _make_eval_step(
    model: SupConEncoder,
    projection_tail: ProjectionTail,
    tau: float,
    distance: str,
    lambda_family,
    lambda_spacegroup,
    lambda_norm,
    has_family: bool,
    has_spacegroup: bool,
):
    """Create a jitted evaluation step for one (unpadded) batch.

    Mirrors ``_make_train_step``'s objective exactly (same terms, same
    weights, including ``norm_penalty``, computed on the projection tail's
    output) so ``val_loss`` reflects the same thing ``train_loss`` is
    actually optimized for.
    """

    @jax.jit
    def _eval_step(params, batch_x, batch_family, batch_spacegroup):
        r = model.encode_with_params(params["body"], batch_x)
        z = projection_tail.project_with_params(params["tail"], r)
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


def train_supcon(
    model: SupConEncoder,
    projection_tail: ProjectionTail,
    train_db: VAEDatabase,
    val_db: VAEDatabase,
    config: TrainConfig,
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
    """Jointly train a ``SupConEncoder`` body and a ``ProjectionTail``
    (Adam by default, or VeLO -- see ``TrainConfig.optimizer``) and return
    per-epoch loss history -- the SupCon loss is computed on the projection
    tail's output, not the body's raw representation (see the module
    docstring). Both ``model.params``/``projection_tail.params``
    are updated in place.

    Args:
        model: ``SupConEncoder`` instance containing the Flax module and
            mutable parameters -- the body.
        projection_tail: ``ProjectionTail`` instance trained jointly with
            the body; its output is what the SupCon loss actually sees.
        train_db: Training dataset wrapper.
        val_db: Validation dataset wrapper.
        config: Training hyperparameters and execution backend options.
        train_family_ids: Integer family class ids (shape ``(n_train,)``),
            aligned row-for-row with ``train_db``. Together with
            ``val_family_ids``, activates the family-level SupCon term
            (weighted by ``lambda_family``). Leave both ``None`` to disable
            it (``mode == "spacegroup_only"``).
        val_family_ids: Integer family class ids aligned with ``val_db``.
        train_spacegroup_ids: Integer spacegroup class ids (shape
            ``(n_train,)``), aligned with ``train_db``. Together with
            ``val_spacegroup_ids``, activates the spacegroup-level SupCon
            term (weighted by ``lambda_spacegroup``). Independent of the
            family term -- no co-occurrence mask needed, unlike the
            aux-head spacegroup classifier in ``vae``/``autoencoder``.
        val_spacegroup_ids: Integer spacegroup class ids aligned with ``val_db``.
        lambda_family: Weight of the family-level term.
        lambda_spacegroup: Weight of the spacegroup-level term.
        lambda_norm: Weight of the embedding-norm regularizer (see
            ``norm_penalty``), applied to the *projection tail's* output
            (the space the SupCon loss is computed in), not the body's raw
            representation. ``0.0`` (default) disables it -- unlike
            ``lambda_family``/``lambda_spacegroup``, always in effect
            (whenever non-zero) regardless of which label ids are given,
            since it doesn't depend on labels at all.
        batching_strategy: ``"random"`` (default -- plain shuffle via
            ``_iter_batches``, unchanged behavior) or ``"balanced"`` (a
            family/spacegroup-stratified sampler, see
            ``dim_red.supcon.sampling.iter_balanced_batches``). Only affects
            *training* batches -- validation always stays ``"random"``
            regardless of this setting.
        batching_family_ids: Integer family id per training row (shape
            ``(n_train,)``), used *only* to group balanced batches --
            required when ``batching_strategy == "balanced"``, independent
            of ``train_family_ids``/``has_family`` (the family SupCon loss
            term can be inactive, e.g. ``mode == "spacegroup_only"``, while
            balanced batching still groups by family; passing this
            separately from ``train_family_ids`` keeps the two independent
            rather than one flag accidentally gating the other).
        batching_spacegroup_ids: Integer spacegroup id per training row,
            always required when ``batching_strategy == "balanced"`` --
            spacegroup stratification is always applied (``batching_S ==
            None`` means "use every spacegroup present per family", not
            "skip stratification"). Same independence from
            ``train_spacegroup_ids`` as ``batching_family_ids``.
        batching_P: Number of families per balanced batch (``None`` -> all
            families present in ``batching_family_ids``). Ignored unless
            ``batching_strategy == "balanced"``.
        batching_K: Number of examples per family per balanced batch.
            Required (positive integer) when ``batching_strategy ==
            "balanced"``; ignored otherwise.
        batching_S: Number of spacegroups to stratify by within each chosen
            family (``None`` -> every spacegroup present for that family,
            split as evenly as possible). Ignored unless ``batching_strategy
            == "balanced"``.

    Returns:
        Dictionary with per-epoch losses, one entry per epoch actually run
        (shorter than ``config.epochs`` if ``config.early_stopping`` stopped
        training early). Always contains ``"train_loss"``/``"val_loss"``
        (the full weighted objective) and ``"train_norm_penalty"``/
        ``"val_norm_penalty"`` (unweighted, always present regardless of
        ``lambda_norm`` -- it's always computed, just not always weighted
        into the total). Also contains ``"train_family_supcon"``/
        ``"val_family_supcon"`` when family ids are given, and
        ``"train_spacegroup_supcon"``/``"val_spacegroup_supcon"`` when
        spacegroup ids are given.

    Raises:
        ValueError: If config values are invalid, neither family nor
            spacegroup ids are given (nothing to contrast on), no matching
            JAX device is found, the family/spacegroup id arguments are
            inconsistent (train/val given without its counterpart), or
            ``batching_strategy`` is invalid or missing its required
            ``batching_*`` arguments.
    """
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
    if config.early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be a positive integer")
    if config.early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta must be >= 0")
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
            "batching_strategy='balanced': train.batch_size (%d) is ignored "
            "for training batches -- effective batch size = P*K = %d*%d = "
            "%d. Validation batches are unaffected (still random shuffle, "
            "using batch_size=%d).",
            config.batch_size,
            effective_P,
            batching_K,
            effective_batch_size,
            config.batch_size,
        )
    else:
        effective_batch_size = config.batch_size

    rng = np.random.default_rng(config.seed)
    batches_per_epoch = int(np.ceil(train_np.shape[0] / effective_batch_size))
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

    train_step = _make_train_step(
        model,
        projection_tail,
        tx,
        config.tau,
        config.distance,
        lambda_family_jax,
        lambda_spacegroup_jax,
        lambda_norm_jax,
        has_family,
        has_spacegroup,
    )
    eval_step = _make_eval_step(
        model,
        projection_tail,
        config.tau,
        config.distance,
        lambda_family_jax,
        lambda_spacegroup_jax,
        lambda_norm_jax,
        has_family,
        has_spacegroup,
    )
    params = {"body": model.params, "tail": projection_tail.params}
    opt_state = tx.init(params)

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
                (train_np, train_family_np, train_spacegroup_np),
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
                (train_np, train_family_np, train_spacegroup_np),
                effective_batch_size,
                rng,
            )
        for batch_x, batch_family, batch_spacegroup in train_batches:
            batch_x_jax = jax.device_put(jnp.asarray(batch_x), device)
            batch_family_jax = jax.device_put(jnp.asarray(batch_family), device)
            batch_spacegroup_jax = jax.device_put(jnp.asarray(batch_spacegroup), device)
            params, opt_state, loss, family_loss, spacegroup_loss, penalty = train_step(
                params, batch_x_jax, batch_family_jax, batch_spacegroup_jax, opt_state
            )
            train_losses.append(float(loss))
            train_ns.append(batch_x.shape[0])
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
        ) = (
            [],
            [],
            [],
            [],
            [],
        )
        for batch_x, batch_family, batch_spacegroup in _iter_batches(
            (val_np, val_family_np, val_spacegroup_np), config.batch_size, rng
        ):
            batch_x_jax = jax.device_put(jnp.asarray(batch_x), device)
            batch_family_jax = jax.device_put(jnp.asarray(batch_family), device)
            batch_spacegroup_jax = jax.device_put(jnp.asarray(batch_spacegroup), device)
            loss, family_loss, spacegroup_loss, penalty = eval_step(
                params, batch_x_jax, batch_family_jax, batch_spacegroup_jax
            )
            val_losses.append(float(loss))
            val_ns.append(batch_x.shape[0])
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
    model.params = params["body"]
    projection_tail.params = params["tail"]
    return history
