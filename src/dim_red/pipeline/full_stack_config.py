"""Config for ``FullStack``: a ``family`` stack plus up to 7 per-crystal-system
expert stacks, each with its own dataset (pyxtal) and hyperparameters. Experts
share an optional ``experts.defaults`` block that is deep-merged under each
expert's own block, so every stack is resolved into a complete ``StackSpec``
before any dataset is built.

Needs jax (it imports ``dim_red.supcon.stack``); ``dim_red.pipeline.config``
itself is untouched and stays importable without it.
"""

import copy
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

from dim_red.pipeline.config import (
    AugmentationConfig,
    EarlyStoppingConfig,
    MaceConfig,
    PyxtalConfig,
    SoapConfig,
    TailTrainSettings,
    TrainSettings,
    _dataclass_from_dict,
    _parse_train_settings,
    load_yaml,
)
from dim_red.supcon.stack import (
    Batching,
    StackConfig,
    stack_config_from_dict,
    stack_config_to_dict,
)
from dim_red.supcon.training import TrainConfig

FAMILY = "family"
EXPERT_NAMES = (
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "trigonal",
    "hexagonal",
    "cubic",
)
STACK_ORDER = (FAMILY,) + EXPERT_NAMES

_MODEL_KINDS = ("supcon", "supcon_mace")
_DISTANCES = ("euclidean", "cosine")
_TOP_KEYS = {"name", "seed", "model_kind", "output_dir", "family", "experts"}
_BLOCK_KEYS = {
    "seed",
    "min_train_rows",
    "data",
    "encoder",
    "projection",
    "contrastive",
    "train",
    "batching",
    "classifier",
    "viz",
}


@dataclass(frozen=True)
class StackDataConfig:
    pyxtal: PyxtalConfig
    augmentation: Optional[AugmentationConfig] = None
    soap: SoapConfig = field(default_factory=SoapConfig)
    mace: MaceConfig = field(default_factory=MaceConfig)


@dataclass(frozen=True)
class StackSpec:
    """One fully resolved stack: its data, its model config, its split ratio,
    its seed and the minimum number of training rows it needs."""

    name: str
    data: StackDataConfig
    model: StackConfig
    val_ratio: float
    seed: int
    min_train_rows: int = 10


@dataclass(frozen=True)
class FullStackConfig:
    name: str
    seed: int
    model_kind: str
    output_dir: str
    stacks: Dict[str, StackSpec]


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _check_distance(name: str, where: str, value: str) -> str:
    if value not in _DISTANCES:
        raise ValueError(
            f"stack {name!r}: {where}.distance must be one of {_DISTANCES}, got {value!r}"
        )
    return value


def _batching(name: str, d: dict) -> Batching:
    batching = Batching(**{k: d[k] for k in ("strategy", "P", "K", "S") if k in d})
    if batching.strategy not in ("random", "balanced"):
        raise ValueError(
            f"stack {name!r}: batching.strategy must be 'random' or 'balanced', "
            f"got {batching.strategy!r}"
        )
    if batching.strategy == "balanced" and not (batching.K and batching.K > 0):
        raise ValueError(f"stack {name!r}: balanced batching needs a positive K")
    return batching


def _train_config(settings, tau: float, distance: str, seed: int) -> TrainConfig:
    return TrainConfig(
        epochs=settings.epochs,
        batch_size=settings.batch_size,
        learning_rate=settings.learning_rate,
        tau=tau,
        distance=distance,
        seed=seed,
        device=settings.device,
        early_stopping=settings.early_stopping.enabled,
        early_stopping_patience=settings.early_stopping.patience,
        early_stopping_min_delta=settings.early_stopping.min_delta,
        early_stopping_restore_best=settings.early_stopping.restore_best_weights,
    )


def _restrict_pyxtal(name: str, pyxtal: dict) -> dict:
    """An expert generates only its own crystal system. Explicit
    ``spacegroups`` are left alone (the user chose them)."""
    if name == FAMILY or pyxtal.get("spacegroups") is not None:
        return pyxtal
    system = name.capitalize()
    families = pyxtal.get("families")
    if families is not None and [f.capitalize() for f in families] != [system]:
        raise ValueError(
            f"stack {name!r}: pyxtal.families must be [{system!r}] (or omitted), "
            f"got {families!r}"
        )
    return {**pyxtal, "families": [system]}


def _check_known_keys(name: str, where: str, d: Dict[str, Any], valid) -> None:
    unknown = set(d) - set(valid)
    if unknown:
        raise ValueError(
            f"stack {name!r}: unknown keys {sorted(unknown)} in {where}; "
            f"valid keys: {sorted(valid)}"
        )


def _field_names(cls) -> set:
    return {f.name for f in dataclasses.fields(cls)}


def _sub(name: str, where: str, block: Dict[str, Any], key: str, valid) -> dict:
    """``block[key]`` as a checked dict (``None``/absent -> ``{}``)."""
    d = block.get(key)
    d = {} if d is None else d
    if not isinstance(d, dict):
        raise ValueError(f"stack {name!r}: {where} must be a mapping, got {d!r}")
    _check_known_keys(name, where, d, valid)
    return d


def _data_dict(name: str, where: str, data_block: dict, key: str, cls) -> dict:
    return _sub(name, where, data_block, key, _field_names(cls))


def _checked_train(name: str, where: str, d: Dict[str, Any], cls) -> Dict[str, Any]:
    d = {} if d is None else d
    _check_known_keys(name, where, d, _field_names(cls))
    es = d.get("early_stopping")
    es = {} if es is None else es
    _check_known_keys(
        name, f"{where}.early_stopping", es, _field_names(EarlyStoppingConfig)
    )
    return d


def parse_stack_spec(name: str, block: Dict[str, Any], run_seed: int) -> StackSpec:
    _check_known_keys(name, "the stack block", block, _BLOCK_KEYS)
    seed = int(block.get("seed", run_seed + STACK_ORDER.index(name)))

    data_block = block.get("data")
    if not isinstance(data_block, dict) or "pyxtal" not in data_block:
        raise ValueError(
            f"stack {name!r}: 'data' must contain a 'pyxtal' block "
            "(FullStack supports data_source pyxtal only)"
        )
    _check_known_keys(
        name, "data", data_block, {"pyxtal", "augmentation", "soap", "mace"}
    )
    pyxtal_dict = _data_dict(name, "data.pyxtal", data_block, "pyxtal", PyxtalConfig)
    pyxtal = _dataclass_from_dict(
        PyxtalConfig, _restrict_pyxtal(name, dict(pyxtal_dict))
    )
    if pyxtal.seed is None:
        pyxtal = dataclasses.replace(pyxtal, seed=seed)
    augmentation = (
        _dataclass_from_dict(
            AugmentationConfig,
            _data_dict(
                name,
                "data.augmentation",
                data_block,
                "augmentation",
                AugmentationConfig,
            ),
        )
        if data_block.get("augmentation") is not None
        else None
    )
    data = StackDataConfig(
        pyxtal=pyxtal,
        augmentation=augmentation,
        soap=_dataclass_from_dict(
            SoapConfig, _data_dict(name, "data.soap", data_block, "soap", SoapConfig)
        ),
        mace=_dataclass_from_dict(
            MaceConfig, _data_dict(name, "data.mace", data_block, "mace", MaceConfig)
        ),
    )

    encoder = _sub(
        name, "encoder", block, "encoder", {"encoder_hidden_dim", "latent_dim"}
    )
    missing = {"encoder_hidden_dim", "latent_dim"} - set(encoder)
    if missing:
        raise ValueError(f"stack {name!r}: encoder block needs {sorted(missing)}")
    projection = _sub(
        name,
        "projection",
        block,
        "projection",
        {"projection_dim", "projection_hidden_dim"},
    )
    contrastive = _sub(
        name, "contrastive", block, "contrastive", {"tau", "distance", "lambda_norm"}
    )
    distance = _check_distance(
        name, "contrastive", contrastive.get("distance", "cosine")
    )
    classifier = _sub(name, "classifier", block, "classifier", {"hidden_dim", "train"})
    viz = _sub(
        name,
        "viz",
        block,
        "viz",
        {
            "hidden_dim",
            "viz_dim",
            "tau",
            "distance",
            "lambda_norm",
            "train",
            "batching",
        },
    )
    viz_distance = _check_distance(name, "viz", viz.get("distance", "euclidean"))
    viz_dim = int(viz.get("viz_dim", 2))
    if viz_dim not in (2, 3):
        raise ValueError(f"stack {name!r}: viz.viz_dim must be 2 or 3, got {viz_dim}")

    train_dict = _checked_train(name, "train", block.get("train"), TrainSettings)
    head_default = {k: v for k, v in train_dict.items() if k != "val_ratio"}
    classifier_train = classifier.get("train")
    viz_train = viz.get("train")
    body_settings = _parse_train_settings(TrainSettings, train_dict)
    classifier_settings = _parse_train_settings(
        TailTrainSettings,
        (
            head_default
            if classifier_train is None
            else _checked_train(
                name, "classifier.train", classifier_train, TailTrainSettings
            )
        ),
    )
    viz_settings = _parse_train_settings(
        TailTrainSettings,
        (
            head_default
            if viz_train is None
            else _checked_train(name, "viz.train", viz_train, TailTrainSettings)
        ),
    )

    val_ratio = body_settings.val_ratio
    if not 0 < val_ratio < 1:
        raise ValueError(
            f"stack {name!r}: train.val_ratio must satisfy 0 < val_ratio < 1, "
            f"got {val_ratio!r}"
        )
    min_train_rows = block.get("min_train_rows", 10)
    if (
        isinstance(min_train_rows, bool)
        or not isinstance(min_train_rows, int)
        or min_train_rows < 1
    ):
        raise ValueError(
            f"stack {name!r}: min_train_rows must be an integer >= 1, "
            f"got {min_train_rows!r}"
        )

    batching_keys = {"strategy", "P", "K", "S"}
    model = StackConfig(
        encoder_hidden_dim=list(encoder["encoder_hidden_dim"]),
        latent_dim=int(encoder["latent_dim"]),
        projection_dim=int(projection.get("projection_dim", 128)),
        projection_hidden_dim=projection.get("projection_hidden_dim"),
        body_train=_train_config(
            body_settings, float(contrastive.get("tau", 0.05)), distance, seed
        ),
        classifier_train=_train_config(classifier_settings, 0.1, "euclidean", seed),
        viz_train=_train_config(
            viz_settings, float(viz.get("tau", 0.1)), viz_distance, seed
        ),
        classifier_hidden_dim=classifier.get("hidden_dim", 16),
        viz_hidden_dim=viz.get("hidden_dim"),
        viz_dim=viz_dim,
        lambda_norm=float(contrastive.get("lambda_norm", 0.0)),
        viz_lambda_norm=float(viz.get("lambda_norm", 0.0)),
        body_batching=_batching(
            name, _sub(name, "batching", block, "batching", batching_keys)
        ),
        viz_batching=_batching(
            name, _sub(name, "viz.batching", viz, "batching", batching_keys)
        ),
        seed=seed,
    )
    return StackSpec(
        name=name,
        data=data,
        model=model,
        val_ratio=val_ratio,
        seed=seed,
        min_train_rows=min_train_rows,
    )


def full_stack_config_from_dict(d: Dict[str, Any]) -> FullStackConfig:
    unknown = set(d) - _TOP_KEYS
    if unknown:
        raise ValueError(
            f"unknown top-level keys {sorted(unknown)}; valid keys: {sorted(_TOP_KEYS)}"
        )
    model_kind = d.get("model_kind", "supcon")
    if model_kind not in _MODEL_KINDS:
        raise ValueError(
            f"model_kind must be one of {_MODEL_KINDS}, got {model_kind!r}"
        )
    seed = int(d.get("seed", 42))

    experts = dict(d.get("experts") or {})
    defaults = experts.pop("defaults", None) or {}
    bad = set(experts) - set(EXPERT_NAMES)
    if bad:
        raise ValueError(
            f"unknown expert name(s) {sorted(bad)}; valid names: {list(EXPERT_NAMES)}"
        )

    blocks: Dict[str, dict] = {}
    if d.get("family") is not None:
        blocks[FAMILY] = dict(d["family"])
    for name in EXPERT_NAMES:
        if name in experts:
            blocks[name] = _deep_merge(defaults, experts[name] or {})
    if not blocks:
        raise ValueError(
            "config defines no stacks: set 'family' and/or at least one expert "
            "under 'experts'"
        )

    stacks = {
        name: parse_stack_spec(name, block, seed) for name, block in blocks.items()
    }
    if model_kind == "supcon_mace":
        for name, spec in stacks.items():
            if not spec.data.mace.checkpoint_path:
                raise ValueError(
                    f"stack {name!r}: model_kind supcon_mace needs "
                    "data.mace.checkpoint_path"
                )
    return FullStackConfig(
        name=str(d.get("name", "full-stack")),
        seed=seed,
        model_kind=model_kind,
        output_dir=str(d.get("output_dir", "runs")),
        stacks=stacks,
    )


def load_full_stack_config(path: Union[str, Path]) -> FullStackConfig:
    return full_stack_config_from_dict(load_yaml(path))


# -- resolved (fully expanded) dicts, used for what is written on disk -------
def stack_spec_to_dict(spec: StackSpec) -> dict:
    d = dataclasses.asdict(spec)
    d["model"] = stack_config_to_dict(spec.model)
    return json.loads(json.dumps(d))


def stack_spec_from_dict(d: dict) -> StackSpec:
    data = d["data"]
    return StackSpec(
        name=d["name"],
        data=StackDataConfig(
            pyxtal=PyxtalConfig(**data["pyxtal"]),
            augmentation=(
                AugmentationConfig(**data["augmentation"])
                if data.get("augmentation")
                else None
            ),
            soap=SoapConfig(**data["soap"]),
            mace=MaceConfig(**data["mace"]),
        ),
        model=stack_config_from_dict(d["model"]),
        val_ratio=d["val_ratio"],
        seed=d["seed"],
        min_train_rows=d["min_train_rows"],
    )


def full_stack_config_to_dict(config: FullStackConfig) -> dict:
    return {
        "name": config.name,
        "seed": config.seed,
        "model_kind": config.model_kind,
        "output_dir": config.output_dir,
        "stacks": {n: stack_spec_to_dict(s) for n, s in config.stacks.items()},
    }


def full_stack_config_from_resolved_dict(d: dict) -> FullStackConfig:
    return FullStackConfig(
        name=d["name"],
        seed=d["seed"],
        model_kind=d["model_kind"],
        output_dir=d["output_dir"],
        stacks={n: stack_spec_from_dict(s) for n, s in d["stacks"].items()},
    )
