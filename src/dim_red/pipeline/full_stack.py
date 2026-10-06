"""``FullStack``: runs 1 family ``SingleStack`` plus up to 7 per-crystal-system
expert ``SingleStack``s, each completely independent -- its own pyxtal
dataset, split, standardization, config and seed. ``fit_body`` and
``fit_heads`` are separate so several head configurations can be trained on
the same saved bodies. A subset of the configured stacks can be trained (and
more can be added to the same run later).

Layout (``<run_dir>``)::

    config.yaml                       resolved FullStackConfig
    stacks/<name>/config.yaml         resolved StackSpec
                  dataset.extxyz  classes.yaml  embeddings.npz
                  body/   encoder_params, projection_params, stack.yaml, loss_history.csv
                  heads/<heads_name>/   classifier/viz params, heads.yaml, histories,
                                        predictions.npz, viz_embeddings.npz, viz_plot.png
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import yaml

from dim_red.analysis.plotting import plot_reduced_space
from dim_red.pipeline._common import (
    _build_vocab_ids,
    _make_unique_run_dir,
    _save_loss_history,
    _split_indices_grouped,
)
from dim_red.pipeline.full_stack_config import (
    FAMILY,
    STACK_ORDER,
    FullStackConfig,
    StackSpec,
    full_stack_config_from_resolved_dict,
    full_stack_config_to_dict,
    stack_spec_to_dict,
)
from dim_red.pipeline.stack_data import StackDataset, build_stack_dataset
from dim_red.supcon.stack import SingleStack

logger = logging.getLogger("dim_red.pipeline")

History = Dict[str, List[float]]


def _labels_for(name: str, dataset: StackDataset) -> list:
    """The family stack contrasts on crystal systems, an expert on spacegroups."""
    if name == FAMILY:
        return list(dataset.labels)
    return [int(s) for s in dataset.spacegroups]


class FullStack:
    def __init__(
        self,
        run_dir: Union[str, Path],
        config: FullStackConfig,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        self.run_dir = Path(run_dir)
        self.config = config
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else Path(config.output_dir) / "_dataset_cache"
        )

    # -- construction ----------------------------------------------------
    @classmethod
    def create(
        cls, config: FullStackConfig, cache_dir: Optional[Union[str, Path]] = None
    ) -> "FullStack":
        run_dir = _make_unique_run_dir(Path(config.output_dir), config.name)
        full_stack = cls(run_dir, config, cache_dir)
        full_stack._write_run_config()
        return full_stack

    @classmethod
    def open(
        cls,
        run_dir: Union[str, Path],
        config: Optional[FullStackConfig] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> "FullStack":
        run_dir = Path(run_dir)
        if config is None:
            with open(run_dir / "config.yaml") as f:
                config = full_stack_config_from_resolved_dict(yaml.safe_load(f))
        full_stack = cls(run_dir, config, cache_dir)
        full_stack._write_run_config()  # records stacks added through `config`
        return full_stack

    def _write_run_config(self) -> None:
        """Writes the resolved config. If one is already recorded, its run-level
        fields and the specs of already-trained stacks are kept (that is what
        was trained); stacks not recorded yet, or recorded but untrained, take
        ``self.config``'s spec. ``self.config`` is replaced by the merged result."""
        path = self.run_dir / "config.yaml"
        resolved = full_stack_config_to_dict(self.config)
        if path.exists():
            with open(path) as f:
                previous = yaml.safe_load(f) or {}
            if previous.get("stacks") is not None:
                merged = dict(resolved["stacks"])
                for name, spec in previous["stacks"].items():
                    if name not in merged or self._stack_dir(name).exists():
                        merged[name] = spec
                resolved = {
                    **{
                        k: previous[k]
                        for k in ("name", "seed", "model_kind", "output_dir")
                    },
                    "stacks": {
                        n: merged[n]
                        for n in sorted(
                            merged,
                            key=lambda n: (
                                STACK_ORDER.index(n) if n in STACK_ORDER else 99
                            ),
                        )
                    },
                }
                self.config = full_stack_config_from_resolved_dict(resolved)
        with open(path, "w") as f:
            yaml.safe_dump(resolved, f, sort_keys=False)

    # -- helpers ---------------------------------------------------------
    def _select(
        self, stacks: Optional[Sequence[str]], config: FullStackConfig
    ) -> List[str]:
        if stacks is None:
            return list(config.stacks)
        unknown = [s for s in stacks if s not in config.stacks]
        if unknown:
            raise ValueError(
                f"stack(s) {unknown} are not configured; configured stacks: "
                f"{list(config.stacks)}"
            )
        return list(stacks)

    def _stack_dir(self, name: str) -> Path:
        return self.run_dir / "stacks" / name

    @staticmethod
    def _validate(name: str, spec: StackSpec, classes: list, n_train: int) -> None:
        if len(classes) < 2:
            raise ValueError(
                f"stack {name!r}: its dataset has {len(classes)} class(es); at "
                "least 2 are needed"
            )
        if n_train < spec.min_train_rows:
            raise ValueError(
                f"stack {name!r}: only {n_train} training rows "
                f"(min_train_rows={spec.min_train_rows})"
            )

    # -- phase 1 ---------------------------------------------------------
    def fit_body(self, stacks: Optional[Sequence[str]] = None) -> Dict[str, History]:
        """Train encoder + projection of each selected stack (all configured
        stacks by default). Raises ``FileExistsError`` before training anything
        if one of them already has a trained body."""
        names = self._select(stacks, self.config)
        existing = [n for n in names if self._stack_dir(n).exists()]
        if existing:
            raise FileExistsError(
                f"stack(s) {existing} already trained in {self.run_dir}; "
                "select other stacks or use a new run"
            )
        histories = {}
        for name in names:
            histories[name] = self._fit_body_one(name, self.config.stacks[name])
        return histories

    def _fit_body_one(self, name: str, spec: StackSpec) -> History:
        dataset = build_stack_dataset(spec, self.config.model_kind, self.cache_dir)
        classes, y = _build_vocab_ids(_labels_for(name, dataset))
        _, sg_ids = _build_vocab_ids([int(s) for s in dataset.spacegroups])
        train_idx, val_idx = _split_indices_grouped(
            dataset.material_ids, spec.val_ratio, spec.seed
        )
        self._validate(name, spec, classes, len(train_idx))

        X = np.asarray(dataset.X, dtype=np.float32)
        stack = SingleStack(X.shape[1], spec.model)
        history = stack.fit_body(
            X[train_idx],
            X[val_idx],
            y[train_idx],
            y[val_idx],
            sub_train=sg_ids[train_idx] if name == FAMILY else None,
        )
        logger.info(
            "Stack %r: body trained on %d rows (%d classes), %d epochs",
            name,
            len(train_idx),
            len(classes),
            len(history["train_loss"]),
        )

        final_dir = self._stack_dir(name)
        stack_dir = final_dir.parent / f".{name}.tmp"
        shutil.rmtree(stack_dir, ignore_errors=True)
        stack_dir.mkdir(parents=True)
        try:
            self._write_body(
                stack_dir,
                name,
                spec,
                stack,
                history,
                dataset,
                X,
                y,
                sg_ids,
                val_idx,
                classes,
            )
            os.replace(stack_dir, final_dir)
        except BaseException:
            shutil.rmtree(stack_dir, ignore_errors=True)
            raise
        return history

    @staticmethod
    def _write_body(
        stack_dir, name, spec, stack, history, dataset, X, y, sg_ids, val_idx, classes
    ) -> None:
        stack.save_body(stack_dir / "body")
        _save_loss_history(stack_dir / "body" / "loss_history.csv", history)
        shutil.copy(dataset.structures_path, stack_dir / "dataset.extxyz")
        split = np.full(len(X), "train", dtype="<U5")
        split[val_idx] = "val"
        np.savez(
            stack_dir / "embeddings.npz",
            embeddings=stack.encode(X),
            features=X,
            labels=np.asarray([str(v) for v in _labels_for(name, dataset)]),
            label_ids=y,
            spacegroups=np.asarray(dataset.spacegroups, dtype=np.int64),
            spacegroup_ids=sg_ids,
            material_ids=np.asarray(dataset.material_ids),
            split=split,
            feature_mean=dataset.feature_mean,
            feature_std=dataset.feature_std,
        )
        with open(stack_dir / "classes.yaml", "w") as f:
            yaml.safe_dump(
                {
                    "role": "family" if name == FAMILY else "spacegroup",
                    "classes": [str(c) if name == FAMILY else int(c) for c in classes],
                },
                f,
                sort_keys=False,
            )
        with open(stack_dir / "config.yaml", "w") as f:
            yaml.safe_dump(stack_spec_to_dict(spec), f, sort_keys=False)

    # -- phase 2 ---------------------------------------------------------
    def fit_heads(
        self,
        heads_name: str,
        stacks: Optional[Sequence[str]] = None,
        config: Optional[FullStackConfig] = None,
    ) -> Dict[str, Dict[str, History]]:
        """Train classification + visualization tails on the saved, frozen
        bodies. ``config`` (optional) supplies different head hyperparameters;
        by default the run's own are used. Several ``heads_name``s can coexist
        on the same body."""
        cfg = config if config is not None else self.config
        names = self._select(stacks, cfg)
        for name in names:
            if not (self._stack_dir(name) / "body" / "stack.yaml").exists():
                raise FileNotFoundError(
                    f"stack {name!r} has no trained body in {self.run_dir}: run "
                    "fit_body first"
                )
            if (self._stack_dir(name) / "heads" / heads_name).exists():
                raise FileExistsError(
                    f"heads {heads_name!r} already exist for stack {name!r}"
                )
        return {
            name: self._fit_heads_one(name, heads_name, cfg.stacks[name])
            for name in names
        }

    def _fit_heads_one(
        self, name: str, heads_name: str, spec: StackSpec
    ) -> Dict[str, History]:
        stack_dir = self._stack_dir(name)
        with np.load(stack_dir / "embeddings.npz") as npz:
            emb = {k: npz[k] for k in npz.files}
        with open(stack_dir / "classes.yaml") as f:
            classes = yaml.safe_load(f)["classes"]
        split = emb["split"]
        train = np.flatnonzero(split == "train")
        val = np.flatnonzero(split == "val")
        X, y, sg_ids = emb["features"], emb["label_ids"], emb["spacegroup_ids"]

        stack = SingleStack.load_body(stack_dir / "body")
        histories = stack.fit_heads(
            X[train],
            X[val],
            y[train],
            y[val],
            n_classes=len(classes),
            sub_train=sg_ids[train] if name == FAMILY else None,
            config=spec.model,
        )

        final_dir = stack_dir / "heads" / heads_name
        heads_dir = final_dir.parent / f".{heads_name}.tmp"
        shutil.rmtree(heads_dir, ignore_errors=True)
        heads_dir.mkdir(parents=True)
        try:
            self._write_heads(
                heads_dir, name, heads_name, classes, stack, histories, emb, split, y, X
            )
            os.replace(heads_dir, final_dir)
        except BaseException:
            shutil.rmtree(heads_dir, ignore_errors=True)
            raise
        return histories

    @staticmethod
    def _write_heads(
        heads_dir, name, heads_name, classes, stack, histories, emb, split, y, X
    ) -> None:
        stack.save_heads(heads_dir)
        _save_loss_history(
            heads_dir / "classifier_loss_history.csv", histories["classifier"]
        )
        _save_loss_history(
            heads_dir / "viz_loss_history.csv", histories["visualization"]
        )
        np.savez(
            heads_dir / "predictions.npz",
            material_ids=emb["material_ids"],
            split=split,
            label_ids=y,
            probs=stack.predict_proba(X),
        )
        z = stack.visualize(X)
        np.savez(
            heads_dir / "viz_embeddings.npz",
            embeddings=z,
            label_ids=y,
            material_ids=emb["material_ids"],
            split=split,
        )
        if z.shape[1] == 2:
            plot_reduced_space(
                z,
                [str(classes[i]) for i in y],
                title=f"{name} / {heads_name}",
                save_path=str(heads_dir / "viz_plot.png"),
                legend=False,
            )

    # -- loading ---------------------------------------------------------
    def load_stack(self, name: str, heads_name: Optional[str] = None) -> SingleStack:
        stack = SingleStack.load_body(self._stack_dir(name) / "body")
        if heads_name is not None:
            stack.load_heads(self._stack_dir(name) / "heads" / heads_name)
        return stack
