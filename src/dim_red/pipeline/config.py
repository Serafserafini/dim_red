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
_MODEL_KINDS = ("vae", "autoencoder")


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
class TrainSettings:
    """Training hyperparameters, plus the train/val split ratio.

    ``beta`` is ignored when ``RunConfig.model_kind == "autoencoder"`` (no
    KL term to weight -- see ``dim_red.autoencoder.training.TrainConfig``).
    """

    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    beta: float = 1.0
    val_ratio: float = 0.2
    device: str = "cpu"


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
class RunConfig:
    """Fully resolved configuration for a single fetch -> SOAP -> model run.

    Attributes:
        model_kind: Which model to train: ``"vae"`` (default, a
            variational autoencoder trained with a KL term/``beta``) or
            ``"autoencoder"`` (a deterministic autoencoder, no KL/``beta``).
            Both read their architecture from ``vae`` (encoder/decoder
            hidden dims, latent dim, mirror) and their aux-head settings from
            ``aux_heads`` -- identical schema either way, see
            ``dim_red.pipeline.single_run.run_single``.
    """

    crystal_systems: List[str]
    limit_per_system: int
    soap: SoapConfig
    vae: VAEArchConfig
    train: TrainSettings
    aux_heads: AuxHeadsConfig = field(default_factory=AuxHeadsConfig)
    seed: int = 42
    output_dir: str = "runs"
    api_key: Optional[str] = None
    name: Optional[str] = None
    model_kind: str = "vae"

    def __post_init__(self):
        if self.model_kind not in _MODEL_KINDS:
            raise ValueError(
                f"model_kind must be one of {_MODEL_KINDS}, got {self.model_kind!r}"
            )


@dataclass(frozen=True)
class SweepConfig:
    """Generic grid sweep: a single-run-shaped ``base`` config (same nested
    shape ``load_run_config`` reads) plus any number of dotted-path axes in
    ``grid`` to Cartesian-product over.

    Any ``RunConfig`` field can be swept this way -- not just a fixed set of
    named axes -- since each grid key is just a path into that same nested
    dict, e.g. ``"vae.encoder_hidden_dim"``, ``"train.learning_rate"``,
    ``"aux_heads.lambda_family"``, ``"data.crystal_systems"``, or a
    top-level field like ``"seed"``.
    """

    base: Dict[str, Any]
    grid: Dict[str, List[Any]] = field(default_factory=dict)

    @property
    def output_dir(self) -> str:
        return str(self.base.get("output_dir", "runs"))

    @property
    def api_key(self) -> Optional[str]:
        return self.base.get("api_key")


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


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_config_from_dict(d: Dict[str, Any]) -> RunConfig:
    data = d.get("data", {})
    soap = _dataclass_from_dict(SoapConfig, d.get("soap", {}))
    vae = _dataclass_from_dict(VAEArchConfig, d.get("vae", {}))
    train = _dataclass_from_dict(TrainSettings, d.get("train", {}))
    aux_heads = _dataclass_from_dict(AuxHeadsConfig, d.get("aux_heads", {}))
    return RunConfig(
        crystal_systems=list(data["crystal_systems"]),
        limit_per_system=int(data.get("limit_per_system", 15)),
        soap=soap,
        vae=vae,
        train=train,
        aux_heads=aux_heads,
        seed=int(d.get("seed", 42)),
        output_dir=str(d.get("output_dir", "runs")),
        api_key=d.get("api_key"),
        name=d.get("name"),
        model_kind=str(d.get("model", "vae")),
    )


def load_run_config(path: Union[str, Path]) -> RunConfig:
    return run_config_from_dict(load_yaml(path))


def run_config_to_dict(config: RunConfig) -> Dict[str, Any]:
    """Serialize a RunConfig back into the same nested shape ``load_run_config``
    expects, so a saved ``config.yaml`` can be fed straight back in for a rerun.
    """
    return {
        "seed": config.seed,
        "output_dir": config.output_dir,
        "api_key": config.api_key,
        "name": config.name,
        "model": config.model_kind,
        "data": {
            "crystal_systems": config.crystal_systems,
            "limit_per_system": config.limit_per_system,
        },
        "soap": dataclasses.asdict(config.soap),
        "vae": dataclasses.asdict(config.vae),
        "train": dataclasses.asdict(config.train),
        "aux_heads": dataclasses.asdict(config.aux_heads),
    }


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
