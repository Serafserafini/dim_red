"""
Configuration schema and YAML loaders for the unified fetch -> SOAP -> VAE
pipeline, covering both single-run configs and grid-sweep configs.
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
    """VAE architecture: this is sweep axis A (hidden-layer configuration)."""

    encoder_hidden_dim: List[int]
    latent_dim: int
    decoder_hidden_dim: Optional[List[int]] = None
    mirror: bool = True


@dataclass(frozen=True)
class TrainSettings:
    """Training hyperparameters, plus the train/val split ratio."""

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
    """Fully resolved configuration for a single fetch -> SOAP -> VAE run."""

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


@dataclass(frozen=True)
class SweepConfig:
    """Grid over sweep axis A (hidden-layer configs) x axis B (crystal-system
    subsets) x, when auxiliary heads are active, the lambda weight(s) of
    their loss terms; every other setting is shared across the grid.
    """

    crystal_system_sets: List[List[str]]
    hidden_layer_configs: List[List[int]]
    limit_per_system: int
    soap: SoapConfig
    latent_dim: int
    train: TrainSettings
    aux_heads_mode: str = "none"
    lambda_family_values: List[float] = field(default_factory=lambda: [1.0])
    lambda_spacegroup_values: List[float] = field(default_factory=lambda: [1.0])
    head_hidden_dim: int = 16
    seed: int = 42
    output_dir: str = "runs"
    api_key: Optional[str] = None
    mirror: bool = True

    def __post_init__(self):
        if self.aux_heads_mode not in _AUX_HEADS_MODES:
            raise ValueError(
                f"aux_heads_mode must be one of {_AUX_HEADS_MODES}, got {self.aux_heads_mode!r}"
            )


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
    data = d.get("data", {})
    vae = d.get("vae", {})
    aux_heads = d.get("aux_heads", {})
    soap = _dataclass_from_dict(SoapConfig, d.get("soap", {}))
    train = _dataclass_from_dict(TrainSettings, d.get("train", {}))
    return SweepConfig(
        crystal_system_sets=[list(s) for s in data["crystal_system_sets"]],
        hidden_layer_configs=[list(h) for h in vae["hidden_layer_configs"]],
        limit_per_system=int(data.get("limit_per_system", 15)),
        soap=soap,
        latent_dim=int(vae["latent_dim"]),
        train=train,
        aux_heads_mode=str(aux_heads.get("mode", "none")),
        lambda_family_values=[
            float(v) for v in aux_heads.get("lambda_family_values", [1.0])
        ],
        lambda_spacegroup_values=[
            float(v) for v in aux_heads.get("lambda_spacegroup_values", [1.0])
        ],
        head_hidden_dim=int(aux_heads.get("head_hidden_dim", 16)),
        seed=int(d.get("seed", 42)),
        output_dir=str(d.get("output_dir", "runs")),
        api_key=d.get("api_key"),
        mirror=bool(vae.get("mirror", True)),
    )


def _expand_aux_heads(sweep: SweepConfig) -> List[AuxHeadsConfig]:
    """Expand the auxiliary-heads axis (lambda_family x lambda_spacegroup),
    only over the lambdas actually used by ``sweep.aux_heads_mode``.
    """
    if sweep.aux_heads_mode == "none":
        return [AuxHeadsConfig(mode="none")]
    if sweep.aux_heads_mode == "family_only":
        return [
            AuxHeadsConfig(
                mode="family_only",
                lambda_family=lf,
                head_hidden_dim=sweep.head_hidden_dim,
            )
            for lf in sweep.lambda_family_values
        ]
    # aux_heads_mode == "family_and_spacegroup" (only remaining valid value).
    return [
        AuxHeadsConfig(
            mode="family_and_spacegroup",
            lambda_family=lf,
            lambda_spacegroup=lsg,
            head_hidden_dim=sweep.head_hidden_dim,
        )
        for lf in sweep.lambda_family_values
        for lsg in sweep.lambda_spacegroup_values
    ]


def expand_sweep(sweep: SweepConfig) -> List[RunConfig]:
    """Expand a sweep's grid -- crystal-system sets x hidden-layer configs x,
    when auxiliary heads are active, their lambda weight(s) -- into one
    RunConfig per combination.
    """
    runs = []
    for cs_set in sweep.crystal_system_sets:
        for hidden_dims in sweep.hidden_layer_configs:
            for aux_cfg in _expand_aux_heads(sweep):
                vae_cfg = VAEArchConfig(
                    encoder_hidden_dim=list(hidden_dims),
                    latent_dim=sweep.latent_dim,
                    decoder_hidden_dim=None,
                    mirror=sweep.mirror,
                )
                runs.append(
                    RunConfig(
                        crystal_systems=list(cs_set),
                        limit_per_system=sweep.limit_per_system,
                        soap=sweep.soap,
                        vae=vae_cfg,
                        train=sweep.train,
                        aux_heads=aux_cfg,
                        seed=sweep.seed,
                        output_dir=sweep.output_dir,
                        api_key=sweep.api_key,
                        name=None,
                    )
                )
    return runs
