"""``SingleStack``: one encoder + projection tail (trained jointly with the
SupCon loss, phase 1 = ``fit_body``) and one classification tail + one
visualization tail (trained on the frozen encoder's output, phase 2 =
``fit_heads``). The same object serves the family level and every
per-crystal-system expert -- only the data, the label and the config differ.

Arrays and models only: no file layout, no plots, no pipeline imports. The
label is a single integer array: it is the contrastive label of the body, the
class of the classifier and the contrastive label of the visualization tail.
"""

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import jax
import numpy as np
import yaml
from flax import serialization

from dim_red.dataset import FeatureDatabase
from dim_red.supcon.model import SupConEncoder
from dim_red.supcon.tail_training import (
    train_classification_tail,
    train_visualization_tail,
)
from dim_red.supcon.tails import ClassificationTail, ProjectionTail, VisualizationTail
from dim_red.supcon.training import TrainConfig, training_first_phase

History = Dict[str, List[float]]


@dataclass(frozen=True)
class Batching:
    """How training batches are formed (same meaning as
    ``training_first_phase``'s ``batching_*`` arguments)."""

    strategy: str = "random"
    P: Optional[int] = None
    K: Optional[int] = None
    S: Optional[int] = None


@dataclass(frozen=True)
class StackConfig:
    """Everything one ``SingleStack`` needs. ``body_train``/``classifier_train``/
    ``viz_train`` are the loop settings of the three training stages; ``tau``/
    ``distance`` of ``body_train`` and ``viz_train`` are the contrastive
    temperature/similarity of the body and of the visualization tail
    (``classifier_train``'s are ignored by the cross-entropy loss)."""

    encoder_hidden_dim: Sequence[int]
    latent_dim: int
    projection_dim: int
    body_train: TrainConfig
    classifier_train: TrainConfig
    viz_train: TrainConfig
    projection_hidden_dim: Optional[Sequence[int]] = None
    classifier_hidden_dim: Union[int, Sequence[int]] = 16
    viz_hidden_dim: Optional[Sequence[int]] = None
    viz_dim: int = 2
    lambda_norm: float = 0.0
    viz_lambda_norm: float = 0.0
    body_batching: Batching = field(default_factory=Batching)
    viz_batching: Batching = field(default_factory=Batching)
    seed: int = 42


def stack_config_to_dict(config: StackConfig) -> dict:
    """JSON/YAML-friendly dict (lists instead of tuples)."""
    return json.loads(json.dumps(dataclasses.asdict(config)))


def stack_config_from_dict(d: dict) -> StackConfig:
    d = dict(d)
    for key in ("body_train", "classifier_train", "viz_train"):
        d[key] = TrainConfig(**d[key])
    for key in ("body_batching", "viz_batching"):
        d[key] = Batching(**d.get(key, {}))
    return StackConfig(**d)


def _batching_kwargs(batching: Batching, y_train, sub_train) -> dict:
    balanced = batching.strategy == "balanced"
    if balanced and sub_train is None:
        raise ValueError(
            "balanced batching needs sub_train (secondary labels, e.g. "
            "spacegroup ids, used to stratify the batches)"
        )
    return dict(
        batching_strategy=batching.strategy,
        batching_family_ids=np.asarray(y_train) if balanced else None,
        batching_spacegroup_ids=np.asarray(sub_train) if balanced else None,
        batching_P=batching.P,
        batching_K=batching.K,
        batching_S=batching.S,
    )


def _write_params(path: Path, params) -> None:
    path.write_bytes(serialization.to_bytes(params))


def _read_params(path: Path, template, device: str):
    # ``from_bytes`` returns numpy leaves. Place them on the backend the
    # params were trained on (``TrainConfig.device``): left to jax's default
    # device they land on the GPU, whose matmuls use TF32 and give ~1e-3
    # different outputs than the same values on the CPU.
    restored = serialization.from_bytes(template, path.read_bytes())
    return jax.device_put(restored, jax.devices(device)[0])


class SingleStack:
    """Encoder + projection tail (``fit_body``), then classification and
    visualization tails on the frozen encoder's output (``fit_heads``)."""

    def __init__(self, input_dim: int, config: StackConfig):
        self.input_dim = int(input_dim)
        self.config = config
        self.encoder = SupConEncoder(
            input_dim=self.input_dim,
            encoder_hidden_dim=config.encoder_hidden_dim,
            latent_dim=config.latent_dim,
            seed=config.seed,
        )
        self.projection = ProjectionTail(
            input_dim=config.latent_dim,
            hidden_dim=config.projection_hidden_dim or [config.latent_dim],
            projection_dim=config.projection_dim,
            seed=config.seed,
        )
        self.classifier: Optional[ClassificationTail] = None
        self.visualizer: Optional[VisualizationTail] = None
        self.n_classes: Optional[int] = None
        self._heads_config: Optional[StackConfig] = None

    # -- phase 1 ---------------------------------------------------------
    def fit_body(self, X_train, X_val, y_train, y_val, sub_train=None) -> History:
        """Train encoder + projection tail jointly with the SupCon loss on
        ``y``. ``sub_train`` (e.g. spacegroup ids) is only needed for
        ``body_batching.strategy == "balanced"``."""
        cfg = self.config
        y_train = np.asarray(y_train)
        y_val = np.asarray(y_val)
        return training_first_phase(
            self.encoder,
            self.projection,
            FeatureDatabase.from_array(X_train),
            FeatureDatabase.from_array(X_val),
            cfg.body_train,
            train_family_ids=y_train,
            val_family_ids=y_val,
            lambda_family=1.0,
            lambda_spacegroup=0.0,
            lambda_norm=cfg.lambda_norm,
            **_batching_kwargs(cfg.body_batching, y_train, sub_train),
        )

    # -- phase 2 ---------------------------------------------------------
    def _build_heads(self, cfg: StackConfig, n_classes: int) -> None:
        self.classifier = ClassificationTail(
            input_dim=self.config.latent_dim,
            hidden_dim=cfg.classifier_hidden_dim,
            n_classes=n_classes,
            seed=cfg.seed,
        )
        self.visualizer = VisualizationTail(
            input_dim=self.config.latent_dim,
            hidden_dim=cfg.viz_hidden_dim or [self.config.latent_dim],
            output_dim=cfg.viz_dim,
            seed=cfg.seed,
        )
        self.n_classes = int(n_classes)
        self._heads_config = cfg

    def fit_heads(
        self,
        X_train,
        X_val,
        y_train,
        y_val,
        n_classes: int,
        sub_train=None,
        config: Optional[StackConfig] = None,
    ) -> Dict[str, History]:
        """Train the classification and visualization tails on the frozen
        encoder's output (the projection tail is not used). ``config``
        overrides the head hyperparameters (``classifier_*``, ``viz_*``,
        ``seed``); encoder fields always come from the stack's own config."""
        cfg = config if config is not None else self.config
        y_train = np.asarray(y_train)
        y_val = np.asarray(y_val)
        r_train = self.encode(X_train)
        r_val = self.encode(X_val)
        self._build_heads(cfg, n_classes)

        classifier_history = train_classification_tail(
            self.classifier,
            r_train,
            r_val,
            cfg.classifier_train,
            train_labels=y_train,
            val_labels=y_val,
        )
        visualization_history = train_visualization_tail(
            self.visualizer,
            r_train,
            r_val,
            cfg.viz_train,
            train_family_ids=y_train,
            val_family_ids=y_val,
            lambda_family=1.0,
            lambda_spacegroup=0.0,
            lambda_norm=cfg.viz_lambda_norm,
            **_batching_kwargs(cfg.viz_batching, y_train, sub_train),
        )
        return {
            "classifier": classifier_history,
            "visualization": visualization_history,
        }

    # -- inference -------------------------------------------------------
    def encode(self, X) -> np.ndarray:
        return np.asarray(self.encoder.encode(X))

    def _require_heads(self) -> None:
        if self.classifier is None or self.visualizer is None:
            raise RuntimeError(
                "heads are not trained or loaded: call fit_heads/load_heads"
            )

    def predict_proba(self, X) -> np.ndarray:
        self._require_heads()
        logits = np.asarray(self.classifier.classify(self.encode(X)))
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def visualize(self, X) -> np.ndarray:
        self._require_heads()
        return np.asarray(self.visualizer.project(self.encode(X)))

    # -- persistence -----------------------------------------------------
    def save_body(self, directory) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        _write_params(directory / "encoder_params.msgpack", self.encoder.params)
        _write_params(directory / "projection_params.msgpack", self.projection.params)
        with open(directory / "stack.yaml", "w") as f:
            yaml.safe_dump(
                {
                    "input_dim": self.input_dim,
                    "config": stack_config_to_dict(self.config),
                },
                f,
                sort_keys=False,
            )

    @classmethod
    def load_body(cls, directory) -> "SingleStack":
        directory = Path(directory)
        with open(directory / "stack.yaml") as f:
            meta = yaml.safe_load(f)
        stack = cls(meta["input_dim"], stack_config_from_dict(meta["config"]))
        stack.encoder.params = _read_params(
            directory / "encoder_params.msgpack",
            stack.encoder.params,
            stack.config.body_train.device,
        )
        stack.projection.params = _read_params(
            directory / "projection_params.msgpack",
            stack.projection.params,
            stack.config.body_train.device,
        )
        return stack

    def save_heads(self, directory) -> None:
        self._require_heads()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        _write_params(directory / "classifier_params.msgpack", self.classifier.params)
        _write_params(directory / "viz_params.msgpack", self.visualizer.params)
        with open(directory / "heads.yaml", "w") as f:
            yaml.safe_dump(
                {
                    "n_classes": self.n_classes,
                    "config": stack_config_to_dict(self._heads_config),
                },
                f,
                sort_keys=False,
            )

    def load_heads(self, directory) -> None:
        directory = Path(directory)
        with open(directory / "heads.yaml") as f:
            meta = yaml.safe_load(f)
        self._build_heads(stack_config_from_dict(meta["config"]), meta["n_classes"])
        heads = self._heads_config
        self.classifier.params = _read_params(
            directory / "classifier_params.msgpack",
            self.classifier.params,
            heads.classifier_train.device,
        )
        self.visualizer.params = _read_params(
            directory / "viz_params.msgpack",
            self.visualizer.params,
            heads.viz_train.device,
        )
