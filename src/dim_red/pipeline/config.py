"""
Configuration schema and YAML loaders for the unified fetch -> SOAP -> VAE
pipeline, covering both single-run configs and grid-sweep configs.
"""

from __future__ import annotations

import copy
import dataclasses
import itertools
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

logger = logging.getLogger("dim_red.pipeline")

_AUX_HEADS_MODES = ("none", "family_only", "family_and_spacegroup")
_SUPCON_MODES = ("family_only", "spacegroup_only", "family_and_spacegroup")
_SUPCON_DISTANCES = ("euclidean", "cosine")
_MODEL_KINDS = ("vae", "autoencoder", "supcon")
_DATA_SOURCES = ("fetch", "pyxtal")
_OPTIMIZER_KINDS = ("adam", "velo")


@dataclass(frozen=True)
class SoapConfig:
    """SOAP hyperparameters shared by every run (average is fixed to "outer"
    by the pipeline itself, since a single global descriptor per structure is
    what the VAE consumes).
    """

    r_cut: float = 5.0
    n_max: int = 4
    l_max: int = 3
    sigma: float = 0.5
    element_agnostic: bool = False
    normalize_distances: bool = False
    species: Optional[List[str]] = None

    def as_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments for ``dim_red.soap.compute_soap`` (minus ``average``)."""
        return {
            "r_cut": self.r_cut,
            "n_max": self.n_max,
            "l_max": self.l_max,
            "sigma": self.sigma,
            "element_agnostic": self.element_agnostic,
            "normalize_distances": self.normalize_distances,
            "species": self.species,
        }


@dataclass(frozen=True)
class FetchConfig:
    """Config for ``RunConfig.data_source == "fetch"``: queries Materials
    Project for real structures. Symmetric with ``PyxtalConfig``'s role as
    the data-source-specific config block for ``data_source == "pyxtal"``.

    Attributes:
        crystal_systems: Crystal systems to fetch (as accepted by
            ``dim_red.fetch.fetch_structures_by_crystal_system``). Required
            (non-empty) when actually used as ``RunConfig.fetch``; YAML
            configs must set it explicitly for ``data_source: fetch``, same
            strictness as before this block existed.
        limit_per_system: Max structures fetched per crystal system.
        api_key: Materials Project API key (falls back to ``MP_API_KEY``).
    """

    crystal_systems: List[str] = field(default_factory=list)
    limit_per_system: int = 15
    api_key: Optional[str] = None


@dataclass(frozen=True)
class PyxtalConfig:
    """Config for ``RunConfig.data_source == "pyxtal"``: builds the dataset
    with ``dim_red.generate`` (synthetic, symmetry-valid structures) instead
    of fetching from Materials Project. Mirrors
    ``dim_red.generate.GenerationConfig``'s fields one-to-one, kept as an
    independent (pyxtal-free) dataclass here so ``dim_red.pipeline.config``
    stays importable without ``pyxtal`` installed -- the actual
    ``dim_red.generate`` import happens lazily in
    ``dim_red.pipeline.dataset_cache``, only when a run actually uses this
    data source.

    Exactly one of ``structures_per_spacegroup``/``structures_per_family``
    must be set, same rule as ``GenerationConfig``; this isn't validated here
    (kept dependency-free) but will raise when the run actually builds its
    dataset.

    Attributes:
        seed: Seed for reproducible generation (species selection, the
            "random" distribution mode, and pyxtal's own RNG). If ``None``
            (default), falls back to ``RunConfig.seed``. Set this explicitly
            to pin the generated dataset independently of ``RunConfig.seed``
            -- e.g. when sweeping other hyperparameters (including
            ``RunConfig.seed`` itself, for repeated-seed training runs) while
            keeping every run on the exact same dataset.
    """

    families: Optional[List[str]] = None
    spacegroups: Optional[List[int]] = None
    structures_per_spacegroup: Optional[int] = None
    structures_per_family: Optional[int] = None
    distribution: str = "uniform"
    n_species: int = 1
    species_pool: Optional[List[str]] = None
    candidate_num_ions: Optional[List[int]] = None
    factor: float = 1.1
    max_count: int = 5
    seed: Optional[int] = None


@dataclass(frozen=True)
class AugmentationConfig:
    """Config for ``RunConfig.augmentation``: optional structure-level data
    augmentation applied to every fetched/generated structure before SOAP
    featurization -- positional jitter (simulated thermal noise, small random
    displacements of every atom) and/or vacancy removal (randomly deleting
    atoms), each independently probability-gated. ``None`` (``RunConfig``'s
    default) disables augmentation entirely, current behavior unchanged.
    Applies uniformly regardless of ``data_source`` -- both ``fetch`` and
    ``pyxtal`` produce a plain ``List[Atoms]`` before SOAP.

    Mirrors ``dim_red.augmentation.AugmentationConfig``'s fields one-to-one,
    kept as an independent dataclass here (like ``PyxtalConfig`` mirrors
    ``GenerationConfig``) so ``dim_red.pipeline.config`` doesn't pull in
    ``dim_red.augmentation``'s import chain (which imports ``dim_red.fetch``,
    and so ``mp_api``/``pymatgen``, unconditionally) just to parse a config --
    the actual ``dim_red.augmentation`` import happens in
    ``dim_red.pipeline.dataset_cache``, which already depends on those
    unconditionally regardless of ``data_source``.

    Every augmented copy is guaranteed to have at least one of jitter/vacancy
    actually applied to it -- see ``dim_red.augmentation.AugmentationConfig``'s
    docstring, including the corollary for when only one mechanism is
    configured at all.

    Attributes:
        n_augmented: Number of augmented copies generated per structure.
        keep_original: Whether the unmodified structure is also kept
            alongside its augmented copies.
        jitter_probability: Probability, per augmented copy, that positional
            jitter (thermal noise) is applied to it at all.
        jitter_std: Standard deviation (Angstroms) of the Gaussian noise
            added to atomic positions when jitter is applied.
        vacancy_probability: Probability, per augmented copy, that vacancy
            removal is applied to it at all.
        vacancy_atom_probability: Probability that any individual atom is
            removed, once a copy has been selected for vacancy removal.
        max_vacancies: Maximum number of atoms removed per augmented copy
            when vacancy removal applies. ``None`` (default) leaves it
            uncapped, short of the structure's own floor of 1 remaining atom.
        supercell_radius: If set (Angstroms), every structure is first
            expanded into a supercell large enough to fit a sphere of this
            radius before any jitter/vacancy is applied -- meant for unit
            cells too small to meaningfully augment (e.g. pyxtal can
            generate cells holding as few as 1-2 atoms). ``None`` (default)
            skips this step entirely, current behavior unchanged. See
            ``dim_red.augmentation.make_supercell_for_radius``.
        seed: Seed for reproducible augmentation. If ``None`` (default),
            falls back to ``RunConfig.seed`` -- same convention as
            ``PyxtalConfig.seed``.
    """

    n_augmented: int = 1
    keep_original: bool = True
    jitter_probability: float = 0.5
    jitter_std: float = 0.05
    vacancy_probability: float = 0.0
    vacancy_atom_probability: float = 0.05
    max_vacancies: Optional[int] = None
    supercell_radius: Optional[float] = None
    seed: Optional[int] = None

    def __post_init__(self):
        if self.n_augmented < 0:
            raise ValueError("augmentation.n_augmented must be >= 0")
        if not 0.0 <= self.jitter_probability <= 1.0:
            raise ValueError("augmentation.jitter_probability must be in [0, 1]")
        if not 0.0 <= self.vacancy_probability <= 1.0:
            raise ValueError("augmentation.vacancy_probability must be in [0, 1]")
        if not 0.0 <= self.vacancy_atom_probability <= 1.0:
            raise ValueError("augmentation.vacancy_atom_probability must be in [0, 1]")
        if self.jitter_std < 0:
            raise ValueError("augmentation.jitter_std must be >= 0")
        if self.max_vacancies is not None and self.max_vacancies < 0:
            raise ValueError("augmentation.max_vacancies must be >= 0")
        if self.supercell_radius is not None and self.supercell_radius <= 0:
            raise ValueError("augmentation.supercell_radius must be > 0")


@dataclass(frozen=True)
class VAEArchConfig:
    """Encoder/decoder architecture: this is sweep axis A (hidden-layer
    configuration). Shared by both model kinds (``RunConfig.model_kind``) --
    a VAE and a plain Autoencoder built from the same ``encoder_hidden_dim``/
    ``latent_dim``/``decoder_hidden_dim``/``mirror`` differ only in how the
    latent code is produced (sampled vs. deterministic) and trained, not in
    this architecture shape. Kept as ``vae`` in configs/dotted-paths for
    backward compatibility with existing sweep configs.
    """

    encoder_hidden_dim: List[int]
    latent_dim: int
    decoder_hidden_dim: Optional[List[int]] = None
    mirror: bool = True


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Early-stopping settings, shared uniformly across all three model
    kinds (``vae``/``autoencoder``/``supcon``) since every training loop's
    history always has a ``"val_loss"`` key -- the monitored metric here,
    not configurable in this first pass.

    Mirrors ``vae.training.TrainConfig``/``autoencoder.training.TrainConfig``/
    ``supcon.training.TrainConfig``'s ``early_stopping*`` fields one-to-one;
    ``pipeline.single_run.run_single`` reads this block and populates those
    fields on whichever ``TrainConfig`` the active ``model_kind`` uses.

    Attributes:
        enabled: If True, stop training once ``val_loss`` hasn't improved by
            more than ``min_delta`` for ``patience`` consecutive epochs.
            Disabled by default (current behavior unchanged).
        patience: Consecutive non-improving epochs tolerated before stopping.
        min_delta: Minimum decrease in ``val_loss`` counted as an improvement.
        restore_best_weights: If True (default), the trained model's params
            are the best-``val_loss`` epoch's rather than necessarily the
            last epoch trained -- whether training stopped early or ran the
            full ``TrainSettings.epochs``.
    """

    enabled: bool = False
    patience: int = 10
    min_delta: float = 0.0
    restore_best_weights: bool = True

    def __post_init__(self):
        if self.patience <= 0:
            raise ValueError("early_stopping.patience must be a positive integer")
        if self.min_delta < 0:
            raise ValueError("early_stopping.min_delta must be >= 0")


@dataclass(frozen=True)
class TrainSettings:
    """Training hyperparameters, plus the train/val split ratio.

    ``beta`` is ignored when ``RunConfig.model_kind == "autoencoder"`` (no
    KL term to weight -- see ``dim_red.autoencoder.training.TrainConfig``).

    Attributes:
        optimizer: ``"adam"`` (default) -- a plain, fast ``optax.adam(learning_rate)``
            -- or ``"velo"`` -- ``learned_optimization``'s pretrained VeLO
            meta-learned optimizer, whose ``num_steps``-dependent setup and
            pretrained-hypernetwork checkpoint load cost real, fixed time
            (several seconds or more, plus a network call attempting to
            resolve Google Cloud credentials that can hang far longer on
            some networks) before training even starts. Same field/semantics
            across ``vae``/``autoencoder``/``supcon`` phase-1 training and
            (via ``TailTrainSettings.optimizer``) phase-2 tail training.
    """

    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    beta: float = 1.0
    val_ratio: float = 0.2
    device: str = "cpu"
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)

    def __post_init__(self):
        if self.optimizer not in _OPTIMIZER_KINDS:
            raise ValueError(
                f"train.optimizer must be one of {_OPTIMIZER_KINDS}, got "
                f"{self.optimizer!r}"
            )


@dataclass(frozen=True)
class AuxHeadsConfig:
    """Optional auxiliary classification heads on the VAE latent ``z``.

    Attributes:
        mode: One of ``"none"`` (plain VAE, default -- current behavior,
            unchanged), ``"family_only"`` (adds a small family-
            classification head), or ``"family_and_spacegroup"`` (adds
            both heads, with the spacegroup head conditioned on family via
            masking -- see ``dim_red.vae.model.apply_family_mask``).
        lambda_family: Weight of the family cross-entropy term. Ignored
            when ``mode == "none"``.
        lambda_spacegroup: Weight of the (family-masked) spacegroup
            cross-entropy term. Ignored unless ``mode ==
            "family_and_spacegroup"``.
        head_hidden_dim: Hidden width of each auxiliary head's single
            hidden layer.
    """

    mode: str = "none"
    lambda_family: float = 1.0
    lambda_spacegroup: float = 1.0
    head_hidden_dim: int = 16

    def __post_init__(self):
        if self.mode not in _AUX_HEADS_MODES:
            raise ValueError(
                f"aux_heads.mode must be one of {_AUX_HEADS_MODES}, got {self.mode!r}"
            )


@dataclass(frozen=True)
class SupConConfig:
    """Config for ``RunConfig.model_kind == "supcon"``: a two-level (family
    and/or spacegroup) Supervised Contrastive loss, computed on a projection
    tail's output (Khosla et al. 2020's ``z = Proj(Enc(x))``, see
    ``dim_red.supcon.tails.ProjectionTail``/``dim_red.supcon.training``) --
    no classifier heads involved at this stage, unlike ``AuxHeadsConfig``
    (whose ``mode`` this mirrors the shape of). Classification/visualization
    tails are trained separately, after this run's body is frozen -- see
    ``dim_red.pipeline.config.TailTrainConfig``.

    Attributes:
        mode: Which label level(s) to contrast on: ``"family_only"``,
            ``"spacegroup_only"``, or ``"family_and_spacegroup"`` (default).
            Unlike ``AuxHeadsConfig.mode``, ``"spacegroup_only"`` is valid
            here -- the two SupCon terms are independent (no family ->
            spacegroup masking needed), so there's no reason a spacegroup
            term would require a family term to also be active.
        lambda_family: Weight of the family-level SupCon term. Ignored when
            ``mode == "spacegroup_only"``.
        lambda_spacegroup: Weight of the spacegroup-level SupCon term.
            Ignored when ``mode == "family_only"``.
        tau: Temperature dividing similarities before the softmax inside the
            SupCon loss.
        distance: Which similarity ``dim_red.supcon.training.supcon_loss``
            computes: ``"euclidean"`` (default -- negative squared Euclidean
            distance, the original formulation here, unbounded scale) or
            ``"cosine"`` (cosine similarity between L2-normalized ``z``
            vectors, bounded to ``[-1, 1]``). Forwarded as-is to
            ``dim_red.supcon.training.TrainConfig.distance``.
        lambda_norm: Weight of the embedding-norm regularizer (mean squared
            L2 norm of the batch's projected ``z``), added to the total loss
            alongside the family/spacegroup terms -- unlike those, it's not
            gated by ``mode`` (it doesn't depend on labels at all, so it's
            always in effect whenever it's non-zero). ``0.0`` (default)
            disables it, matching the pre-regularizer behavior exactly. See
            ``dim_red.supcon.training.norm_penalty`` for why this matters:
            with ``distance == "euclidean"`` (default), similarity has no
            built-in scale normalization, so nothing otherwise discourages
            the projection from inflating ``z``'s norm without bound;
            largely redundant with ``distance == "cosine"``.
        projection_dim: Size of the space the SupCon loss is actually
            computed in (Khosla et al. 2020 default: 128) -- separate from
            ``vae.latent_dim``, which stays the body's own representation
            width (saved to ``embeddings.npz``, reused by every downstream
            tail). The projection tail is discarded after phase 1 in the
            original paper; this codebase still saves its trained params
            (``projection_params.msgpack``) for reproducibility.
        projection_hidden_dim: Widths of hidden layers in the projection
            tail's MLP. ``None`` (default) resolves to a single hidden
            layer matching the body's own ``latent_dim``.
    """

    mode: str = "family_and_spacegroup"
    lambda_family: float = 1.0
    lambda_spacegroup: float = 1.0
    tau: float = 0.1
    distance: str = "euclidean"
    lambda_norm: float = 0.0
    projection_dim: int = 128
    projection_hidden_dim: Optional[List[int]] = None

    def __post_init__(self):
        if self.mode not in _SUPCON_MODES:
            raise ValueError(
                f"supcon.mode must be one of {_SUPCON_MODES}, got {self.mode!r}"
            )
        if self.distance not in _SUPCON_DISTANCES:
            raise ValueError(
                f"supcon.distance must be one of {_SUPCON_DISTANCES}, got "
                f"{self.distance!r}"
            )
        if self.projection_dim <= 0:
            raise ValueError("supcon.projection_dim must be a positive integer")


@dataclass(frozen=True)
class BalancedBatchingParams:
    """Parameters for ``BatchingConfig.strategy == "balanced"``: batches are
    built as ``P`` families x ``K`` examples per family (optionally further
    stratified into ``S`` spacegroups per family, ``K / S`` examples each),
    instead of a plain random shuffle -- see ``dim_red.supcon.sampling``.

    Attributes:
        P: Number of crystal families included per batch. ``None`` (default)
            means all families present in the training split.
        K: Number of examples per family per batch. Required (must be a
            positive integer) when ``BatchingConfig.strategy == "balanced"``.
        S: Number of distinct spacegroups sampled per family per batch.
            ``None`` (default) uses *every* spacegroup present for that
            family, splitting ``K`` as evenly as possible across all of them
            -- not "skip stratification". Set an explicit integer to sample
            only that many spacegroups per family instead.
    """

    P: Optional[int] = None
    K: Optional[int] = None
    S: Optional[int] = None


_BATCHING_STRATEGIES = ("random", "balanced")


@dataclass(frozen=True)
class BatchingConfig:
    """How ``dim_red.supcon.training.train_supcon`` forms training
    mini-batches. Only meaningful for ``RunConfig.model_kind == "supcon"``
    (ignored otherwise, same treatment as ``aux_heads``/``supcon`` being
    ignored for the model kinds they don't apply to).

    Attributes:
        strategy: ``"random"`` (default -- a plain shuffle, unchanged
            behavior) or ``"balanced"`` (a family/spacegroup-stratified
            sampler, see ``BalancedBatchingParams``). Validation batches
            always stay ``"random"`` regardless of this setting -- only
            training batches are affected.
        balanced_params: Required (with a valid ``K``) when
            ``strategy == "balanced"``; ignored for ``"random"``.
    """

    strategy: str = "random"
    balanced_params: BalancedBatchingParams = field(
        default_factory=BalancedBatchingParams
    )

    def __post_init__(self):
        if self.strategy not in _BATCHING_STRATEGIES:
            raise ValueError(
                f"batching.strategy must be one of {_BATCHING_STRATEGIES}, "
                f"got {self.strategy!r}"
            )
        if self.strategy == "balanced":
            k = self.balanced_params.K
            if k is None or k <= 0:
                raise ValueError(
                    "batching.balanced_params.K must be a positive integer "
                    "when batching.strategy == 'balanced'"
                )


@dataclass(frozen=True)
class RunConfig:
    """Fully resolved configuration for a single dataset -> SOAP -> model run.

    Attributes:
        model_kind: Which model to train: ``"vae"`` (default, a
            variational autoencoder trained with a KL term/``beta``),
            ``"autoencoder"`` (a deterministic autoencoder, no KL/``beta``),
            or ``"supcon"`` (an encoder-only model with no decoder/KL, trained
            with a Supervised Contrastive loss on family/spacegroup labels
            instead of reconstruction -- see ``dim_red.supcon``). All three
            read their encoder architecture from ``vae`` (``encoder_hidden_dim``,
            ``latent_dim`` -- ``decoder_hidden_dim``/``mirror`` are ignored by
            ``"supcon"``, same treatment ``"autoencoder"`` gives ``beta``).
            ``"vae"``/``"autoencoder"`` read their aux-head settings from
            ``aux_heads``; ``"supcon"`` reads its loss settings from
            ``supcon`` instead (``aux_heads`` is ignored for it), and its
            training-batch sampling strategy from ``batching`` (ignored for
            ``"vae"``/``"autoencoder"``, which always use a plain shuffle).
            See ``dim_red.pipeline.single_run.run_single``.
        data_source: How the dataset (before SOAP) is built: ``"fetch"``
            (default -- the ``fetch`` config block queries Materials
            Project) or ``"pyxtal"`` (the ``pyxtal`` config block builds a
            synthetic dataset with ``dim_red.generate`` instead). Exactly
            one of ``fetch``/``pyxtal`` is required, matching
            ``data_source`` -- symmetric config blocks for the two data
            sources. See ``dim_red.pipeline.dataset_cache``.
        fetch: Required when ``data_source == "fetch"``, otherwise unused.
        pyxtal: Required when ``data_source == "pyxtal"``, otherwise unused.
        augmentation: Optional structure-level data augmentation (thermal-
            noise-style positional jitter and/or vacancy removal) applied
            after the dataset is built, before SOAP. ``None`` (default)
            disables it, current behavior unchanged. Applies regardless of
            ``data_source``. See ``AugmentationConfig``.
        tails: Optionally auto-train a classification and/or visualization
            tail (``dim_red.pipeline.tail_training.train_tail``) right after
            this run's body finishes phase-1 training -- the same entry
            point ``dimred-train-tail`` uses manually, just invoked
            automatically. ``None`` (default) disables it, current behavior
            unchanged. Ignored for ``model_kind != "supcon"``, same
            treatment ``aux_heads``/``supcon``/``batching`` get for the
            model kinds they don't apply to. See ``AutoTailsConfig``.
    """

    soap: SoapConfig
    vae: VAEArchConfig
    train: TrainSettings
    aux_heads: AuxHeadsConfig = field(default_factory=AuxHeadsConfig)
    supcon: SupConConfig = field(default_factory=SupConConfig)
    batching: BatchingConfig = field(default_factory=BatchingConfig)
    seed: int = 42
    output_dir: str = "runs"
    name: Optional[str] = None
    model_kind: str = "vae"
    data_source: str = "fetch"
    fetch: Optional[FetchConfig] = None
    pyxtal: Optional[PyxtalConfig] = None
    augmentation: Optional[AugmentationConfig] = None
    tails: Optional["AutoTailsConfig"] = None

    def __post_init__(self):
        if self.model_kind not in _MODEL_KINDS:
            raise ValueError(
                f"model_kind must be one of {_MODEL_KINDS}, got {self.model_kind!r}"
            )
        if self.data_source not in _DATA_SOURCES:
            raise ValueError(
                f"data_source must be one of {_DATA_SOURCES}, got {self.data_source!r}"
            )
        if self.data_source == "fetch" and self.fetch is None:
            raise ValueError("data_source='fetch' requires a 'fetch' config block.")
        if self.data_source == "pyxtal" and self.pyxtal is None:
            raise ValueError("data_source='pyxtal' requires a 'pyxtal' config block.")


@dataclass(frozen=True)
class SweepConfig:
    """Generic grid sweep: a single-run-shaped ``base`` config (same nested
    shape ``load_run_config`` reads) plus any number of dotted-path axes in
    ``grid`` to Cartesian-product over.

    Any ``RunConfig`` field can be swept this way -- not just a fixed set of
    named axes -- since each grid key is just a path into that same nested
    dict, e.g. ``"vae.encoder_hidden_dim"``, ``"train.learning_rate"``,
    ``"aux_heads.lambda_family"``, ``"fetch.crystal_systems"``, or a
    top-level field like ``"seed"``.
    """

    base: Dict[str, Any]
    grid: Dict[str, List[Any]] = field(default_factory=dict)

    @property
    def output_dir(self) -> str:
        return str(self.base.get("output_dir", "runs"))

    @property
    def api_key(self) -> Optional[str]:
        return self.base.get("fetch", {}).get("api_key")


def _dataclass_from_dict(cls, d: Dict[str, Any]):
    """Build a dataclass instance from a dict, ignoring unknown keys (with a
    warning, to surface config typos without hard-failing).
    """
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(d) - known
    if unknown:
        logger.warning(
            "Ignoring unknown %s config keys: %s", cls.__name__, sorted(unknown)
        )
    return cls(**{k: v for k, v in d.items() if k in known})


def _parse_train_settings(cls, d: Dict[str, Any]):
    """Build a ``TrainSettings``/``TailTrainSettings`` instance from a
    ``train:``-shaped dict, handling the nested ``early_stopping:`` sub-block
    the same way for both (every training-loop settings dataclass has an
    ``early_stopping: EarlyStoppingConfig`` field).
    """
    early_stopping = _dataclass_from_dict(
        EarlyStoppingConfig, d.get("early_stopping", {})
    )
    settings = _dataclass_from_dict(
        cls, {k: v for k, v in d.items() if k != "early_stopping"}
    )
    return dataclasses.replace(settings, early_stopping=early_stopping)


def _parse_classification_tail_config(
    d: Dict[str, Any]
) -> Optional["ClassificationTailConfig"]:
    """Parse a ``classification:`` block, shared by ``RunConfig.tails`` and
    ``TailTrainConfig``."""
    return (
        _dataclass_from_dict(ClassificationTailConfig, d["classification"])
        if "classification" in d
        else None
    )


def _parse_visualization_tail_config(
    d: Dict[str, Any]
) -> Optional["VisualizationTailConfig"]:
    """Parse a ``visualization:`` block (including its nested ``batching:``
    sub-block), shared by ``RunConfig.tails`` and ``TailTrainConfig``."""
    visualization_dict = d.get("visualization")
    if visualization_dict is None:
        return None
    batching_dict = visualization_dict.get("batching", {})
    batching = BatchingConfig(
        strategy=str(batching_dict.get("strategy", "random")),
        balanced_params=_dataclass_from_dict(
            BalancedBatchingParams, batching_dict.get("balanced_params", {})
        ),
    )
    visualization = _dataclass_from_dict(
        VisualizationTailConfig,
        {k: v for k, v in visualization_dict.items() if k != "batching"},
    )
    return dataclasses.replace(visualization, batching=batching)


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_config_from_dict(d: Dict[str, Any]) -> RunConfig:
    data_source = str(d.get("data_source", "fetch"))
    soap = _dataclass_from_dict(SoapConfig, d.get("soap", {}))
    vae = _dataclass_from_dict(VAEArchConfig, d.get("vae", {}))
    train = _parse_train_settings(TrainSettings, d.get("train", {}))
    aux_heads = _dataclass_from_dict(AuxHeadsConfig, d.get("aux_heads", {}))
    supcon = _dataclass_from_dict(SupConConfig, d.get("supcon", {}))
    batching_dict = d.get("batching", {})
    batching = BatchingConfig(
        strategy=str(batching_dict.get("strategy", "random")),
        balanced_params=_dataclass_from_dict(
            BalancedBatchingParams, batching_dict.get("balanced_params", {})
        ),
    )
    pyxtal_config = (
        _dataclass_from_dict(PyxtalConfig, d["pyxtal"]) if "pyxtal" in d else None
    )
    augmentation_config = (
        _dataclass_from_dict(AugmentationConfig, d["augmentation"])
        if "augmentation" in d
        else None
    )
    tails_dict = d.get("tails")
    tails_config = (
        AutoTailsConfig(
            classification=_parse_classification_tail_config(tails_dict),
            visualization=_parse_visualization_tail_config(tails_dict),
            train=_parse_train_settings(TailTrainSettings, tails_dict.get("train", {})),
        )
        if tails_dict is not None
        else None
    )

    fetch_config: Optional[FetchConfig] = None
    if "fetch" in d:
        fetch_dict = d["fetch"]
        # "crystal_systems" is only required when it's actually used
        # (data_source == "fetch"); a "fetch" block left over in a sweep's
        # base config for a pyxtal-mode run doesn't need it set.
        crystal_systems = (
            list(fetch_dict["crystal_systems"])
            if data_source == "fetch"
            else list(fetch_dict.get("crystal_systems", []))
        )
        fetch_config = FetchConfig(
            crystal_systems=crystal_systems,
            limit_per_system=int(fetch_dict.get("limit_per_system", 15)),
            api_key=fetch_dict.get("api_key"),
        )
    elif data_source == "fetch":
        raise KeyError("crystal_systems")

    return RunConfig(
        soap=soap,
        vae=vae,
        train=train,
        aux_heads=aux_heads,
        supcon=supcon,
        batching=batching,
        seed=int(d.get("seed", 42)),
        output_dir=str(d.get("output_dir", "runs")),
        name=d.get("name"),
        model_kind=str(d.get("model", "vae")),
        data_source=data_source,
        fetch=fetch_config,
        pyxtal=pyxtal_config,
        augmentation=augmentation_config,
        tails=tails_config,
    )


def load_run_config(path: Union[str, Path]) -> RunConfig:
    return run_config_from_dict(load_yaml(path))


def run_config_to_dict(config: RunConfig) -> Dict[str, Any]:
    """Serialize a RunConfig back into the same nested shape ``load_run_config``
    expects, so a saved ``config.yaml`` can be fed straight back in for a rerun.
    """
    result = {
        "seed": config.seed,
        "output_dir": config.output_dir,
        "name": config.name,
        "model": config.model_kind,
        "data_source": config.data_source,
        "soap": dataclasses.asdict(config.soap),
        "vae": dataclasses.asdict(config.vae),
        "train": dataclasses.asdict(config.train),
        "aux_heads": dataclasses.asdict(config.aux_heads),
        "supcon": dataclasses.asdict(config.supcon),
        "batching": dataclasses.asdict(config.batching),
    }
    if config.fetch is not None:
        result["fetch"] = dataclasses.asdict(config.fetch)
    if config.pyxtal is not None:
        result["pyxtal"] = dataclasses.asdict(config.pyxtal)
    if config.augmentation is not None:
        result["augmentation"] = dataclasses.asdict(config.augmentation)
    if config.tails is not None:
        tails_dict: Dict[str, Any] = {"train": dataclasses.asdict(config.tails.train)}
        if config.tails.classification is not None:
            tails_dict["classification"] = dataclasses.asdict(
                config.tails.classification
            )
        if config.tails.visualization is not None:
            tails_dict["visualization"] = dataclasses.asdict(config.tails.visualization)
        result["tails"] = tails_dict
    return result


def load_sweep_config(path: Union[str, Path]) -> SweepConfig:
    d = load_yaml(path)
    base = d.get("base", {})
    grid = d.get("grid", {})
    if not grid:
        logger.warning(
            "Sweep config %s has an empty 'grid'; expand_sweep will produce a "
            "single run from 'base' alone",
            path,
        )
    return SweepConfig(base=base, grid={k: list(v) for k, v in grid.items()})


def _set_dotted(d: Dict[str, Any], path: str, value: Any) -> None:
    """Set a nested dict's value at dotted ``path`` (e.g. ``"vae.latent_dim"``),
    creating intermediate dicts as needed. Mirrors the nested shape
    ``run_config_from_dict`` reads, so any grid key can override any single-run
    config field.
    """
    parts = path.split(".")
    cur = d
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def flatten_config_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested config dict (as produced by ``run_config_to_dict``)
    into ``{dotted_path: leaf_value}`` pairs -- the inverse of ``_set_dotted``.
    Only ``dict`` values are recursed into; lists/None/scalars are leaves.
    """
    flat: Dict[str, Any] = {}
    for key, value in d.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_config_dict(value, dotted))
        else:
            flat[dotted] = value
    return flat


def expand_sweep(sweep: SweepConfig) -> List[RunConfig]:
    """Expand a sweep's grid -- the Cartesian product of every dotted-path
    axis in ``sweep.grid`` -- into one ``RunConfig`` per combination, each
    built by overriding ``sweep.base`` at those paths. An empty grid produces
    a single run from ``base`` alone.
    """
    if not sweep.grid:
        return [run_config_from_dict(copy.deepcopy(sweep.base))]

    keys = list(sweep.grid)
    value_lists = [sweep.grid[k] for k in keys]
    runs = []
    for combo in itertools.product(*value_lists):
        d = copy.deepcopy(sweep.base)
        for key, value in zip(keys, combo):
            _set_dotted(d, key, value)
        runs.append(run_config_from_dict(d))
    return runs


# --- Phase 2: tail training (dim_red.pipeline.tail_training) ---------------
#
# Deliberately independent of RunConfig/SweepConfig above: this targets an
# already-completed `model: supcon` run directory rather than building a
# dataset from scratch, so it gets its own top-level YAML shape and loader
# rather than being nested under RunConfig.

_TAIL_KINDS = ("classification", "visualization")
_CLASSIFICATION_TAIL_MODES = ("family_only", "family_and_spacegroup")


@dataclass(frozen=True)
class ClassificationTailConfig:
    """Config for ``TailTrainConfig.tail_kind == "classification"``: trains
    a ``dim_red.supcon.tails.ClassificationTail`` on a frozen supcon run's
    saved representations, via cross-entropy. Mirrors ``AuxHeadsConfig``'s
    3-value ``mode`` restriction (not ``SupConConfig``'s) -- the spacegroup
    head here is family-masked via ``dim_red.supcon.tails.apply_family_mask``,
    so an unconditioned ``"spacegroup_only"`` isn't meaningful, same
    restriction ``AuxHeadsConfig.mode`` already has and for the same reason.

    Attributes:
        mode: ``"family_only"`` or ``"family_and_spacegroup"`` (default).
        lambda_family: Weight of the family cross-entropy term.
        lambda_spacegroup: Weight of the (family-masked) spacegroup
            cross-entropy term. Ignored unless ``mode ==
            "family_and_spacegroup"``.
        head_hidden_dim: Hidden width of each head's single hidden layer.
    """

    mode: str = "family_and_spacegroup"
    lambda_family: float = 1.0
    lambda_spacegroup: float = 1.0
    head_hidden_dim: int = 16

    def __post_init__(self):
        if self.mode not in _CLASSIFICATION_TAIL_MODES:
            raise ValueError(
                f"classification.mode must be one of "
                f"{_CLASSIFICATION_TAIL_MODES}, got {self.mode!r}"
            )


@dataclass(frozen=True)
class VisualizationTailConfig:
    """Config for ``TailTrainConfig.tail_kind == "visualization"``: trains a
    ``dim_red.supcon.tails.VisualizationTail`` on a frozen supcon run's
    saved representations, via the same Supervised Contrastive loss as the
    phase-1 projection tail (see
    ``dim_red.supcon.tail_training.train_visualization_tail``).

    Attributes:
        viz_dim: Output dimensionality -- 2 or 3 (plotted with
            ``dim_red.analysis.plotting.plot_reduced_space``/
            ``plot_reduced_space_3d`` respectively).
        mode: Which label level(s) to contrast on -- same 3-value set as
            ``SupConConfig.mode`` (independent terms, no family ->
            spacegroup masking, unlike ``ClassificationTailConfig.mode``).
        lambda_family: Weight of the family-level term. Ignored when
            ``mode == "spacegroup_only"``.
        lambda_spacegroup: Weight of the spacegroup-level term. Ignored
            when ``mode == "family_only"``.
        tau: Temperature dividing similarities before the softmax.
        distance: ``"euclidean"`` (default) or ``"cosine"`` -- see
            ``dim_red.supcon.training.supcon_loss``.
        lambda_norm: Weight of the embedding-norm regularizer, applied to
            this tail's own output. ``0.0`` (default) disables it.
        hidden_dim: Widths of hidden layers in the visualization MLP.
            ``None`` (default) resolves to a single hidden layer matching
            the body's own representation width.
        batching: How training batches are formed -- same ``BatchingConfig``
            shape/semantics as ``RunConfig.batching`` (``"random"`` default,
            or ``"balanced"`` via ``dim_red.supcon.sampling``).
    """

    viz_dim: int = 2
    mode: str = "family_and_spacegroup"
    lambda_family: float = 1.0
    lambda_spacegroup: float = 1.0
    tau: float = 0.1
    distance: str = "euclidean"
    lambda_norm: float = 0.0
    hidden_dim: Optional[List[int]] = None
    batching: BatchingConfig = field(default_factory=BatchingConfig)

    def __post_init__(self):
        if self.viz_dim not in (2, 3):
            raise ValueError(
                f"visualization.viz_dim must be 2 or 3, got {self.viz_dim!r}"
            )
        if self.mode not in _SUPCON_MODES:
            raise ValueError(
                f"visualization.mode must be one of {_SUPCON_MODES}, got "
                f"{self.mode!r}"
            )
        if self.distance not in _SUPCON_DISTANCES:
            raise ValueError(
                f"visualization.distance must be one of {_SUPCON_DISTANCES}, "
                f"got {self.distance!r}"
            )


@dataclass(frozen=True)
class TailTrainSettings:
    """Training-loop mechanics for phase 2, mirroring the relevant subset of
    ``RunConfig.train`` (``TrainSettings``) -- no ``beta``/``val_ratio``:
    phase 2 reuses phase 1's exact train/val split (see
    ``dim_red.pipeline.tail_training``) rather than resplitting.

    ``optimizer`` is the same field/semantics as ``TrainSettings.optimizer``
    -- ``"adam"`` (default) or ``"velo"`` -- and matters even more here than
    in phase 1, since phase-2 tails are tiny single-hidden-layer MLPs where
    VeLO's fixed setup cost dominates the actual training time.
    """

    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    device: str = "cpu"
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)

    def __post_init__(self):
        if self.optimizer not in _OPTIMIZER_KINDS:
            raise ValueError(
                f"train.optimizer must be one of {_OPTIMIZER_KINDS}, got "
                f"{self.optimizer!r}"
            )


@dataclass(frozen=True)
class AutoTailsConfig:
    """Config for ``RunConfig.tails``: automatically train one or both tails
    on a completed ``model_kind == "supcon"`` run's frozen body, right after
    phase-1 training finishes -- the same
    ``dim_red.pipeline.tail_training.train_tail`` entry point
    ``dimred-train-tail`` uses, just invoked automatically instead of as a
    separate manual command. Ignored for ``model_kind != "supcon"``, same
    treatment ``aux_heads``/``supcon``/``batching`` already get for the model
    kinds they don't apply to.

    Attributes:
        classification: Auto-train a classification tail when set (``None``
            default disables it).
        visualization: Auto-train a visualization tail when set (``None``
            default disables it). A run/sweep can set both to get a
            classification tail AND a visualization tail out of a single
            ``dimred-run``/``dimred-sweep`` invocation.
        train: Training-loop mechanics shared by whichever tail(s) are
            enabled -- same shape as ``TailTrainConfig.train``.
    """

    classification: Optional[ClassificationTailConfig] = None
    visualization: Optional[VisualizationTailConfig] = None
    train: TailTrainSettings = field(default_factory=TailTrainSettings)


@dataclass(frozen=True)
class TailTrainConfig:
    """Fully resolved configuration for phase 2: freeze an already-trained
    supcon run's body and train exactly one tail (classification or
    visualization) on top of it -- see ``dim_red.pipeline.tail_training``.

    Attributes:
        tail_kind: ``"classification"`` or ``"visualization"`` -- which tail
            to train. Exactly one of ``classification``/``visualization``
            must be set, matching this.
        run_dir: Path to a completed ``model: supcon`` run directory (must
            contain at least ``config.yaml``/``embeddings.npz`` --
            ``dim_red.pipeline.inference.load_run_embeddings``'s required
            set; ``dataset.extxyz``/``model_params.msgpack`` aren't needed
            for tail training at all). ``None`` (default) leaves it unset
            in the config itself -- the
            ``dimred-train-tail <config> <run_dir>`` console script (and its
            ``python -m dim_red.pipeline.cli --train-tail <config>
            --train-tail-run <run_dir>`` flag-based form) always takes the
            run directory as a separate argument and overrides whatever is
            here, so a single tail-training config can be reused across many
            runs without editing it each time; set this directly only for
            programmatic use of ``dim_red.pipeline.tail_training.train_tail``
            without going through the CLI. ``train_tail`` raises if it's
            still ``None`` by the time training actually starts.
        classification: Required when ``tail_kind == "classification"``.
        visualization: Required when ``tail_kind == "visualization"``.
        train: Training-loop mechanics.
        seed: Seed for the tail's own parameter initialization.
        output_subdir: Directory name under ``<run_dir>/tails/`` this tail's
            artifacts are written to. ``None`` (default) uses ``tail_kind``
            itself.
    """

    tail_kind: str
    run_dir: Optional[str] = None
    classification: Optional[ClassificationTailConfig] = None
    visualization: Optional[VisualizationTailConfig] = None
    train: TailTrainSettings = field(default_factory=TailTrainSettings)
    seed: int = 42
    output_subdir: Optional[str] = None

    def __post_init__(self):
        if self.tail_kind not in _TAIL_KINDS:
            raise ValueError(
                f"tail_kind must be one of {_TAIL_KINDS}, got {self.tail_kind!r}"
            )
        if self.tail_kind == "classification" and self.classification is None:
            raise ValueError(
                "tail_kind='classification' requires a 'classification' config block."
            )
        if self.tail_kind == "visualization" and self.visualization is None:
            raise ValueError(
                "tail_kind='visualization' requires a 'visualization' config block."
            )


def tail_train_config_from_dict(d: Dict[str, Any]) -> TailTrainConfig:
    """Build a ``TailTrainConfig`` from a parsed YAML dict (see
    ``load_tail_train_config``)."""
    tail_kind = str(d.get("tail_kind", "classification"))
    train = _parse_train_settings(TailTrainSettings, d.get("train", {}))
    classification = _parse_classification_tail_config(d)
    visualization = _parse_visualization_tail_config(d)

    run_dir = d.get("run_dir")
    return TailTrainConfig(
        run_dir=str(run_dir) if run_dir is not None else None,
        tail_kind=tail_kind,
        classification=classification,
        visualization=visualization,
        train=train,
        seed=int(d.get("seed", 42)),
        output_subdir=d.get("output_subdir"),
    )


def load_tail_train_config(path: Union[str, Path]) -> TailTrainConfig:
    return tail_train_config_from_dict(load_yaml(path))


def tail_train_config_to_dict(config: TailTrainConfig) -> Dict[str, Any]:
    """Serialize a ``TailTrainConfig`` back into the same nested shape
    ``load_tail_train_config`` expects, so a saved ``tail_config.yaml`` can
    be fed straight back in for a rerun.
    """
    result = {
        "run_dir": config.run_dir,
        "tail_kind": config.tail_kind,
        "train": dataclasses.asdict(config.train),
        "seed": config.seed,
        "output_subdir": config.output_subdir,
    }
    if config.classification is not None:
        result["classification"] = dataclasses.asdict(config.classification)
    if config.visualization is not None:
        result["visualization"] = dataclasses.asdict(config.visualization)
    return result
