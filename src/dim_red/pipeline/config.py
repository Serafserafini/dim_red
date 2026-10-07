"""
Configuration schema and YAML loaders for the unified structures -> features
-> model pipeline, covering both single-run configs and grid-sweep configs.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

logger = logging.getLogger("dim_red.pipeline")

_AUX_HEADS_MODES = ("none", "family_only", "family_and_spacegroup")
# Label-level/similarity choices of the SupCon-loss visualization tail
# (VisualizationTailConfig), which cgcnn tails still use.
_SUPCON_MODES = ("family_only", "spacegroup_only", "family_and_spacegroup")
_SUPCON_DISTANCES = ("euclidean", "cosine")
# RunConfig is cgcnn-only; supcon/supcon_mace configs use the FullStack
# schema (dim_red.pipeline.full_stack_config).
_MODEL_KINDS = ("cgcnn",)
_DATA_SOURCES = ("fetch", "pyxtal")


@dataclass(frozen=True)
class SoapConfig:
    """SOAP hyperparameters shared by every run (average is fixed to "outer"
    by the pipeline itself, since a single global descriptor per structure is
    what the models consume).
    """

    r_cut: float = 5.0
    n_max: int = 4
    l_max: int = 3
    sigma: float = 0.5
    element_agnostic: bool = False
    species: Optional[List[str]] = None

    def as_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments for ``dim_red.soap.compute_soap`` (minus ``average``)."""
        return {
            "r_cut": self.r_cut,
            "n_max": self.n_max,
            "l_max": self.l_max,
            "sigma": self.sigma,
            "element_agnostic": self.element_agnostic,
            "species": self.species,
        }


@dataclass(frozen=True)
class GraphConfig:
    """CGCNN graph-construction + architecture hyperparameters
    (``RunConfig.model_kind == "cgcnn"`` only) -- used *instead of* ``soap``
    for that model kind, the same "own dataclass, own kwargs helper" pattern
    ``SoapConfig`` establishes.

    ``latent_dim`` is deliberately NOT a field here -- it's read from the
    shared ``encoder:`` block (``encoder.latent_dim``) instead.
    For ``model_kind: cgcnn``, every other ``encoder:`` field is ignored --
    only ``encoder.latent_dim`` is read (the body's output width, saved to
    ``embeddings.npz``, reused by every downstream tail).

    ``max_species`` doubles as both a graph-construction parameter (bounds
    ``dim_red.cgcnn.graph.atoms_list_to_graph_arrays``' species-remapping,
    part of the dataset cache key) and an architecture parameter (sizes
    ``CGCNNEncoder``'s embedding table) -- both call sites read it straight
    off this same field, so they can never drift apart.

    Attributes:
        radius: Cutoff radius (Angstroms) for periodic-boundary-aware
            neighbor search (``dim_red.cgcnn.graph``, via
            ``ase.neighborlist`` -- correct even when ``radius`` exceeds
            half the unit cell's width, unlike a naive minimum-image-
            convention distance matrix). CGCNN's own default (Xie &
            Grossman 2018).
        max_num_nbr: Maximum neighbors kept per atom -- the closest ones
            within ``radius``; atoms with fewer are padded/masked.
        dmin: Lower bound of the Gaussian distance-expansion filter bank.
        dmax: Upper bound of the Gaussian distance-expansion filter bank.
            ``None`` (default) resolves to ``radius``.
        step: Spacing between Gaussian filters -- together with
            ``dmin``/resolved ``dmax`` this sets ``n_gaussian`` (the
            bond/edge feature width), see the ``n_gaussian`` property.
        max_species: Maximum distinct species any one structure may have.
            The model never sees which real chemical element a species is
            -- only a per-structure local species slot (``1..max_species``)
            -- see ``dim_red.cgcnn.graph._local_species_indices``.
        atom_fea_len: Per-atom embedding/hidden width used throughout the
            graph-convolution stack.
        n_conv: Number of stacked gated graph-convolution layers.
        h_fea_len: Hidden width of the post-pooling fully-connected block.
        n_h: Number of post-pooling hidden layers.
    """

    radius: float = 8.0
    max_num_nbr: int = 12
    dmin: float = 0.0
    dmax: Optional[float] = None
    step: float = 0.2
    max_species: int = 10
    atom_fea_len: int = 64
    n_conv: int = 3
    h_fea_len: int = 128
    n_h: int = 1

    @property
    def resolved_dmax(self) -> float:
        """``dmax`` if explicitly set, otherwise ``radius``."""
        return self.dmax if self.dmax is not None else self.radius

    @property
    def n_gaussian(self) -> int:
        """Width of the Gaussian-expanded bond/edge feature vector."""
        return int(round((self.resolved_dmax - self.dmin) / self.step)) + 1

    def graph_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments for
        ``dim_red.cgcnn.graph.atoms_list_to_graph_arrays`` -- the
        graph-construction subset (``radius``/``max_num_nbr``/``dmin``/
        ``dmax``/``step``/``max_species``), not the architecture-only
        fields (``atom_fea_len``/``n_conv``/``h_fea_len``/``n_h``), which
        affect model construction only, not the cached graph arrays --
        mirrors ``SoapConfig.as_kwargs()``'s role in splitting
        feature-computation params from architecture params.
        """
        return {
            "radius": self.radius,
            "max_num_nbr": self.max_num_nbr,
            "dmin": self.dmin,
            "dmax": self.dmax,
            "step": self.step,
            "max_species": self.max_species,
        }

    def __post_init__(self):
        if self.radius <= 0:
            raise ValueError("graph.radius must be > 0")
        if self.max_num_nbr <= 0:
            raise ValueError("graph.max_num_nbr must be a positive integer")
        if self.step <= 0:
            raise ValueError("graph.step must be > 0")
        if self.dmax is not None and self.dmax <= self.dmin:
            raise ValueError("graph.dmax must be > graph.dmin")
        if self.max_species <= 0:
            raise ValueError("graph.max_species must be a positive integer")
        if (
            self.atom_fea_len <= 0
            or self.n_conv <= 0
            or self.h_fea_len <= 0
            or self.n_h <= 0
        ):
            raise ValueError(
                "graph.atom_fea_len/n_conv/h_fea_len/n_h must all be positive integers"
            )


@dataclass(frozen=True)
class MaceConfig:
    """Config for a ``supcon_mace`` FullStack's ``data.mace`` block
    (``dim_red.pipeline.full_stack_config``): a frozen, pretrained MACE
    (equivariant message-passing, 3-body/angular interactions) feature
    extractor -- used *instead of* SOAP for that model kind. ``mace`` never trains: ``checkpoint_path`` points to an
    already-pretrained, already-converted checkpoint, and its architecture
    (and therefore its output width) is whatever that checkpoint says, not
    something this config chooses. The SupCon body trained on top of its
    embeddings is sized by ``encoder:`` as usual.

    See ``dim_red.mace.model.MaceEncoder``/``dim_red.mace.model.load_frozen_checkpoint``.

    Attributes:
        checkpoint_path: Path to a MACE-JAX checkpoint converted from a
            pretrained Torch foundation model (e.g. MACE-MP-0) via
            ``mace_jax``'s own ``mace-jax-from-torch`` CLI, run once outside
            this codebase -- see ``src/dim_red/mace/CLAUDE.md``. Required
            (non-empty) whenever ``model_kind == "supcon_mace"``.
        r_max: Cutoff radius (Angstroms) for neighbor search -- MUST match
            the cutoff the checkpoint was pretrained with.
        pooling: ``"mean"`` (default) or ``"sum"`` -- how per-atom features
            are pooled into one per-structure representation.
    """

    checkpoint_path: str = ""
    r_max: float = 6.0
    pooling: str = "mean"

    def __post_init__(self):
        if self.r_max <= 0:
            raise ValueError("mace.r_max must be > 0")
        if self.pooling not in ("mean", "sum"):
            raise ValueError(
                f"mace.pooling must be 'mean' or 'sum', got {self.pooling!r}"
            )

    def mace_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments for ``dim_red.mace.model.MaceEncoder`` -- mirrors
        ``GraphConfig.graph_kwargs()``'s/``SoapConfig.as_kwargs()``'s role,
        and doubles as the dataset-cache-key hashed payload (see
        ``dim_red.pipeline.dataset_cache._mace_cache_key``).
        """
        return {
            "checkpoint_path": self.checkpoint_path,
            "r_max": self.r_max,
            "pooling": self.pooling,
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
        jitter_std: Standard deviation of the Gaussian noise added to atomic
            positions when jitter is applied, in Angstroms.
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
class EncoderConfig:
    """Encoder architecture: this is sweep axis A (hidden-layer
    configuration). ``cgcnn`` builds its own graph encoder from
    ``GraphConfig`` and reads only ``latent_dim`` from here.

    In YAML this block is ``encoder:``; the old name ``vae:`` is still read
    (and merged) so older configs and saved ``config.yaml`` files keep
    loading.
    """

    encoder_hidden_dim: List[int]
    latent_dim: int


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Early-stopping settings, shared uniformly across every model kind
    since every training loop's history always has a ``"val_loss"`` key --
    the monitored metric here, not configurable in this first pass.

    Mirrors ``supcon.training.TrainConfig``/``cgcnn.training.TrainConfig``'s
    ``early_stopping*`` fields one-to-one;
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

    ``learning_rate`` is Adam's learning rate (the only optimizer).
    """

    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    val_ratio: float = 0.2
    device: str = "cpu"
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)


@dataclass(frozen=True)
class AuxHeadsConfig:
    """Classification heads trained jointly with the cgcnn body (its only
    training objective).

    Attributes:
        mode: One of ``"none"`` (default; invalid for ``cgcnn``),
            ``"family_only"`` (a small family-classification head), or
            ``"family_and_spacegroup"`` (both heads, with the spacegroup
            head conditioned on family via masking -- see
            ``dim_red.cgcnn.model.apply_family_mask``).
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
    """How a visualization tail (``VisualizationTailConfig.batching``) forms
    training mini-batches.

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
    """Fully resolved configuration for a single dataset -> graphs -> CGCNN
    run (``dim_red.pipeline.single_run.run_single``).

    Attributes:
        model_kind: ``"cgcnn"`` (default): a graph-convolutional encoder
            trained jointly with classifier head(s) via cross-entropy on
            family/spacegroup labels, in a single phase; always requires
            ``aux_heads.mode != "none"``; see ``dim_red.cgcnn``. It reads its
            graph-construction + architecture hyperparameters from ``graph``
            (only ``encoder.latent_dim`` is read from ``encoder``) and its
            classification-head settings from ``aux_heads``.
            Every other model is rejected; supcon/supcon_mace configs use the
            FullStack schema (``dim_red.pipeline.full_stack_config``).
        data_source: How the dataset is built: ``"fetch"``
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
            after the dataset is built. ``None`` (default) disables it.
            Applies regardless of ``data_source``. See
            ``AugmentationConfig``.
    """

    soap: SoapConfig
    encoder: EncoderConfig
    train: TrainSettings
    aux_heads: AuxHeadsConfig = field(default_factory=AuxHeadsConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    seed: int = 42
    output_dir: str = "runs"
    name: Optional[str] = None
    model_kind: str = "cgcnn"
    data_source: str = "fetch"
    fetch: Optional[FetchConfig] = None
    pyxtal: Optional[PyxtalConfig] = None
    augmentation: Optional[AugmentationConfig] = None

    def __post_init__(self):
        if self.model_kind not in _MODEL_KINDS:
            raise ValueError(
                f"model_kind must be one of {_MODEL_KINDS}, got "
                f"{self.model_kind!r}; supcon/supcon_mace runs use "
                "dim_red.pipeline.full_stack.FullStack"
            )
        if self.data_source not in _DATA_SOURCES:
            raise ValueError(
                f"data_source must be one of {_DATA_SOURCES}, got {self.data_source!r}"
            )
        if self.data_source == "fetch" and self.fetch is None:
            raise ValueError("data_source='fetch' requires a 'fetch' config block.")
        if self.data_source == "pyxtal" and self.pyxtal is None:
            raise ValueError("data_source='pyxtal' requires a 'pyxtal' config block.")
        if self.model_kind == "cgcnn" and self.aux_heads.mode == "none":
            raise ValueError(
                "model_kind='cgcnn' requires aux_heads.mode != 'none' "
                "(family classification is CGCNN's only training objective)"
            )


@dataclass(frozen=True)
class SweepConfig:
    """Generic grid sweep: a single-run-shaped ``base`` config (same nested
    shape ``load_run_config`` reads) plus any number of dotted-path axes in
    ``grid`` to Cartesian-product over.

    Any ``RunConfig`` field can be swept this way -- not just a fixed set of
    named axes -- since each grid key is just a path into that same nested
    dict, e.g. ``"encoder.encoder_hidden_dim"``, ``"train.learning_rate"``,
    ``"aux_heads.lambda_family"``, ``"fetch.crystal_systems"``, or a
    top-level field like ``"seed"``.
    """

    base: Dict[str, Any]
    grid: Dict[str, List[Any]] = field(default_factory=dict)

    @property
    def output_dir(self) -> str:
        return str(self.base.get("output_dir", "runs"))


# Keys that older configs / saved ``config.yaml`` files may still carry for
# features that no longer exist. Ignored silently (debug log only) so existing
# runs stay loadable without a warning for every one of them.
_DEPRECATED_KEYS = frozenset(
    {
        "normalize_distances",
        "jitter_std_relative",
        "optimizer",  # train/tails.train: VeLO removed, Adam only
        "beta",  # train: VAE-only KL weight
        "decoder_hidden_dim",  # encoder/vae: no decoders any more
        "mirror",
    }
)

# Top-level RunConfig blocks of removed features. Every config.yaml saved by
# run_single before their removal carries "supcon"/"batching"/"mace" (they
# were always serialized, cgcnn runs included), so they are ignored rather
# than rejected -- otherwise no existing cgcnn run could be reloaded
# (dimred-rerun/dimred-apply/dimred-train-tail).
_REMOVED_RUN_BLOCKS = frozenset({"supcon", "batching", "mace"})


def _dataclass_from_dict(cls, d: Dict[str, Any]):
    """Build a dataclass instance from a dict, ignoring unknown keys (with a
    warning, to surface config typos without hard-failing). Keys in
    ``_DEPRECATED_KEYS`` are dropped without a warning.
    """
    known = {f.name for f in dataclasses.fields(cls)}
    deprecated = (set(d) - known) & _DEPRECATED_KEYS
    if deprecated:
        logger.debug(
            "Ignoring deprecated %s config keys: %s", cls.__name__, sorted(deprecated)
        )
    unknown = set(d) - known - _DEPRECATED_KEYS
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
    """Parse a ``TailTrainConfig`` ``classification:`` block.

    Raises:
        ValueError: If the block still asks for the removed single-stage
            ``mode: family_and_spacegroup`` (silently training a family-only
            classifier instead would not be what the config says).
    """
    if "classification" not in d:
        return None
    block = d["classification"] or {}
    if block.get("mode") == "family_and_spacegroup":
        raise ValueError(
            "classification.mode='family_and_spacegroup' (a single head with "
            "a family-masked spacegroup head) was removed; spacegroup "
            "classification is done by the per-family experts of a FullStack "
            "(dim_red.pipeline.full_stack)."
        )
    return _dataclass_from_dict(ClassificationTailConfig, block)


def _parse_visualization_tail_config(
    d: Dict[str, Any]
) -> Optional["VisualizationTailConfig"]:
    """Parse a ``TailTrainConfig`` ``visualization:`` block (including its
    nested ``batching:`` sub-block)."""
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
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} is empty or not a YAML mapping")
    return loaded


def run_config_from_dict(d: Dict[str, Any]) -> RunConfig:
    """Build a cgcnn ``RunConfig`` from a parsed YAML dict.

    Raises:
        ValueError: If ``model`` is anything but ``"cgcnn"`` (supcon/
            supcon_mace configs use the FullStack schema), or if the removed
            top-level ``tails:`` block is present.
    """
    model_kind = str(d.get("model", "cgcnn"))
    if model_kind != "cgcnn":
        raise ValueError(
            f"model must be 'cgcnn' here, got {model_kind!r}; supcon/supcon_mace "
            "configs use the FullStack schema (model_kind: ...) -- see "
            "dim_red.pipeline.full_stack_config"
        )
    if "tails" in d:
        raise ValueError(
            "The top-level 'tails:' block (auto-training tails after a run) was "
            "removed; train cgcnn tails with dimred-train-tail <config> <run_dir>, "
            "and supcon/supcon_mace heads as part of a FullStack "
            "(dim_red.pipeline.full_stack)."
        )
    removed = sorted(_REMOVED_RUN_BLOCKS & set(d))
    if removed:
        logger.debug("Ignoring removed RunConfig blocks: %s", removed)
    data_source = str(d.get("data_source", "fetch"))
    soap = _dataclass_from_dict(SoapConfig, d.get("soap", {}))
    # ``vae:`` is the pre-rename name of this block; merge it under ``encoder:``
    # so older configs, saved config.yaml files and sweep axes keep working.
    encoder = _dataclass_from_dict(
        EncoderConfig, {**d.get("vae", {}), **d.get("encoder", {})}
    )
    train = _parse_train_settings(TrainSettings, d.get("train", {}))
    aux_heads = _dataclass_from_dict(AuxHeadsConfig, d.get("aux_heads", {}))
    graph = _dataclass_from_dict(GraphConfig, d.get("graph", {}))
    pyxtal_config = (
        _dataclass_from_dict(PyxtalConfig, d["pyxtal"]) if "pyxtal" in d else None
    )
    augmentation_config = (
        _dataclass_from_dict(AugmentationConfig, d["augmentation"])
        if "augmentation" in d
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
        encoder=encoder,
        train=train,
        aux_heads=aux_heads,
        graph=graph,
        seed=int(d.get("seed", 42)),
        output_dir=str(d.get("output_dir", "runs")),
        name=d.get("name"),
        model_kind=model_kind,
        data_source=data_source,
        fetch=fetch_config,
        pyxtal=pyxtal_config,
        augmentation=augmentation_config,
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
        "encoder": dataclasses.asdict(config.encoder),
        "train": dataclasses.asdict(config.train),
        "aux_heads": dataclasses.asdict(config.aux_heads),
        "graph": dataclasses.asdict(config.graph),
    }
    if config.fetch is not None:
        result["fetch"] = dataclasses.asdict(config.fetch)
    if config.pyxtal is not None:
        result["pyxtal"] = dataclasses.asdict(config.pyxtal)
    if config.augmentation is not None:
        result["augmentation"] = dataclasses.asdict(config.augmentation)
    return result


def load_sweep_config(path: Union[str, Path]) -> SweepConfig:
    d = load_yaml(path)
    base = d.get("base", {})
    grid = d.get("grid", {})
    if not grid:
        logger.warning(
            "Sweep config %s has an empty 'grid'; the sweep will produce a "
            "single run from 'base' alone",
            path,
        )
    return SweepConfig(base=base, grid={k: list(v) for k, v in grid.items()})


def _set_dotted(d: Dict[str, Any], path: str, value: Any) -> None:
    """Set a nested dict's value at dotted ``path`` (e.g. ``"encoder.latent_dim"``),
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


# --- Phase 2: tail training (dim_red.pipeline.tail_training) ---------------
#
# Deliberately independent of RunConfig/SweepConfig above: this targets an
# already-completed `model: cgcnn` run directory rather than building a
# dataset from scratch, so it gets its own top-level YAML shape and loader
# rather than being nested under RunConfig.

_TAIL_KINDS = ("classification", "visualization")


@dataclass(frozen=True)
class ClassificationTailConfig:
    """Config for ``TailTrainConfig.tail_kind == "classification"``: trains
    a family classifier (a ``dim_red.supcon.tails.ClassificationTail``) on a
    frozen run's saved representations, via cross-entropy.

    Spacegroup classification is not done here: it is handled by the
    per-family experts of a FullStack (``dim_red.pipeline.full_stack``).

    Attributes:
        head_hidden_dim: Hidden width of the head's single hidden layer.
    """

    head_hidden_dim: int = 16


@dataclass(frozen=True)
class VisualizationTailConfig:
    """Config for ``TailTrainConfig.tail_kind == "visualization"``: trains a
    ``dim_red.supcon.tails.VisualizationTail`` on a frozen cgcnn run's
    saved representations, via the same Supervised Contrastive loss as the
    phase-1 projection tail (see
    ``dim_red.supcon.tail_training.train_visualization_tail``).

    Attributes:
        viz_dim: Output dimensionality -- 2 or 3 (plotted with
            ``dim_red.analysis.plotting.plot_reduced_space``/
            ``plot_reduced_space_3d`` respectively).
        mode: Which label level(s) to contrast on: ``"family_only"``,
            ``"spacegroup_only"`` or ``"family_and_spacegroup"``
            (independent terms, no family -> spacegroup masking).
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
        batching: How training batches are formed -- ``"random"`` (default)
            or ``"balanced"`` via ``dim_red.supcon.sampling``; see
            ``BatchingConfig``.
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
    ``RunConfig.train`` (``TrainSettings``) -- no ``val_ratio``: phase 2
    reuses phase 1's exact train/val split (see
    ``dim_red.pipeline.tail_training``) rather than resplitting.
    """

    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    device: str = "cpu"
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)


@dataclass(frozen=True)
class TailTrainConfig:
    """Fully resolved configuration for phase 2: freeze an already-trained
    cgcnn run's body and train exactly one tail (classification or
    visualization) on top of it -- see ``dim_red.pipeline.tail_training``.
    Which ``model_kind``s a given ``tail_kind`` accepts is documented on
    ``dim_red.pipeline.tail_training._TAIL_MODEL_KINDS``.

    Attributes:
        tail_kind: ``"classification"`` or ``"visualization"`` -- which
            tail to train. The matching config block must be set.
        run_dir: Path to a completed run directory whose ``model_kind`` is
            allowed for ``tail_kind`` (must contain at least
            ``config.yaml``/``embeddings.npz`` --
            ``dim_red.pipeline.inference.load_run_embeddings``'s required
            set; ``dataset.extxyz``/``model_params.msgpack`` aren't needed
            for tail training at all). ``None`` (default) leaves it unset
            in the config itself -- the
            ``dimred-train-tail <config> <run_dir>`` console script always
            takes the run directory as a separate argument and overrides
            whatever is here, so a single tail-training config can be reused across many
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
    ``load_tail_train_config``).

    Raises:
        ValueError: If the removed ``hierarchical_supcon:`` block is present
            (or ``tail_kind`` names it).
    """
    if "hierarchical_supcon" in d:
        raise ValueError(
            "The 'hierarchical_supcon' tail was removed; per-family spacegroup "
            "experts are part of a FullStack (dim_red.pipeline.full_stack)."
        )
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
