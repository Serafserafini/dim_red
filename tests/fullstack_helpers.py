"""Helpers for the FullStack wiring tests: tiny *trained* runs built without
pyxtal/SOAP (a fake dataset builder), and a writer of fake FullStack-layout
run directories (no training) for compare/benchmark tests."""

import csv
from pathlib import Path

import numpy as np
import yaml
from ase import Atoms
from ase.io import write

from dim_red.pipeline.full_stack import FullStack
from dim_red.pipeline.full_stack_config import FAMILY, full_stack_config_from_dict
from dim_red.pipeline.stack_data import StackDataset

SYSTEM_SPACEGROUPS = {
    "cubic": [195, 196],
    "tetragonal": [75, 76],
    "hexagonal": [168, 169],
}


def block(learning_rate=1e-3, latent_dim=4, epochs=2, **over):
    b = {
        "data": {
            "pyxtal": {"structures_per_spacegroup": 2},
            "soap": {"element_agnostic": True},
        },
        "encoder": {"encoder_hidden_dim": [8], "latent_dim": latent_dim},
        "projection": {"projection_dim": 5},
        "train": {"epochs": epochs, "batch_size": 8, "learning_rate": learning_rate},
        "classifier": {"hidden_dim": 6},
        "viz": {"hidden_dim": [5]},
        "min_train_rows": 5,
    }
    b.update(over)
    return b


def config_dict(
    output_dir,
    name="fs",
    experts=("cubic", "tetragonal"),
    with_family=True,
    **block_kwargs,
):
    d = {
        "name": name,
        "seed": 3,
        "output_dir": str(output_dir),
        "experts": {"defaults": block(**block_kwargs), **{e: {} for e in experts}},
    }
    if with_family:
        d["family"] = block(**block_kwargs)
    return d


def fake_builder(tmp_path, calls, rows_per_sg=12):
    def build(spec, model_kind, cache_dir):
        calls.append(spec)
        rng = np.random.default_rng(spec.seed)
        if spec.name == FAMILY:
            plan = [
                (system.capitalize(), sg)
                for system, sgs in SYSTEM_SPACEGROUPS.items()
                for sg in sgs
            ]
        else:
            plan = [
                (spec.name.capitalize(), sg) for sg in SYSTEM_SPACEGROUPS[spec.name]
            ]
        X, labels, material_ids, spacegroups = [], [], [], []
        for _, (family, sg) in enumerate(plan):
            center = rng.normal(size=6) * 3.0
            for i in range(rows_per_sg):
                X.append(center + rng.normal(size=6) * 0.3)
                labels.append(family)
                material_ids.append(f"{spec.name}-{sg}-{i}")
                spacegroups.append(sg)
        path = tmp_path / f"{spec.name}-{len(calls)}.extxyz"
        write(str(path), [Atoms("Cu", positions=[[0, 0, 0]])] * len(X), format="extxyz")
        return StackDataset(
            X=np.asarray(X, dtype=np.float32),
            labels=labels,
            material_ids=material_ids,
            spacegroups=spacegroups,
            structures_path=path,
            feature_mean=np.zeros(6, dtype=np.float32),
            feature_std=np.ones(6, dtype=np.float32),
        )

    return build


def make_run(
    tmp_path,
    monkeypatch,
    *,
    output_dir=None,
    name="fs",
    experts=("cubic", "tetragonal"),
    with_family=True,
    heads=("default",),
    **block_kwargs,
) -> FullStack:
    """Train a tiny FullStack run (body + the named heads) on fake datasets."""
    calls = []
    monkeypatch.setattr(
        "dim_red.pipeline.full_stack.build_stack_dataset",
        fake_builder(tmp_path, calls),
    )
    cfg = full_stack_config_from_dict(
        config_dict(
            output_dir if output_dir is not None else tmp_path / "runs",
            name=name,
            experts=experts,
            with_family=with_family,
            **block_kwargs,
        )
    )
    fs = FullStack.create(cfg, cache_dir=tmp_path / "cache")
    fs.fit_body()
    for heads_name in heads:
        fs.fit_heads(heads_name)
    return fs


def _write_loss_history(path, epochs=3, val_loss_final=1.0, aux=False):
    fields = [
        "epoch",
        "train_loss",
        "val_loss",
        "train_family_supcon",
        "val_family_supcon",
    ]
    if aux:
        fields += ["train_family_ce", "val_family_ce"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            val = val_loss_final + (epochs - epoch) * 0.1
            row = {
                "epoch": epoch,
                "train_loss": val + 0.05,
                "val_loss": val,
                "train_family_supcon": val * 0.9,
                "val_family_supcon": val * 0.8,
            }
            if aux:
                row.update(train_family_ce=0.2, val_family_ce=0.25)
            writer.writerow(row)


def write_fake_run(
    run_dir,
    *,
    stacks=("family",),
    hidden_dim=(4,),
    latent_dim=2,
    learning_rate=1e-3,
    val_loss_final=1.0,
    n=8,
    heads=("default",),
    families=("Cubic",),
    model_kind="supcon",
    n_features=None,
    aux=False,
) -> Path:
    """A FullStack-layout run directory with fake (untrained) contents: every
    file ``open_stack`` reads, nothing else. ``heads=()`` writes no heads."""
    run_dir = Path(run_dir)
    rng = np.random.default_rng(0)
    specs = {}
    for stack in stacks:
        spec = {
            "name": stack,
            "seed": 0,
            "val_ratio": 0.25,
            "min_train_rows": 5,
            "data": {"pyxtal": {"families": list(families)}},
            "model": {
                "encoder_hidden_dim": list(hidden_dim),
                "latent_dim": latent_dim,
                "body_train": {
                    "learning_rate": learning_rate,
                    "epochs": 3,
                    "batch_size": 4,
                },
            },
        }
        specs[stack] = spec
        stack_dir = run_dir / "stacks" / stack
        (stack_dir / "body").mkdir(parents=True)
        (stack_dir / "body" / "stack.yaml").write_text("{}\n")
        _write_loss_history(
            stack_dir / "body" / "loss_history.csv",
            val_loss_final=val_loss_final,
            aux=aux,
        )
        (stack_dir / "config.yaml").write_text(yaml.safe_dump(spec))

        if stack == FAMILY:
            classes = list(families)
            labels = np.array([classes[i % len(classes)] for i in range(n)])
            spacegroups = np.array([195 + (i % 2) for i in range(n)], dtype=np.int64)
            label_ids = np.array([i % len(classes) for i in range(n)])
            role = "family"
        else:
            classes = [195, 196]
            spacegroups = np.array([classes[i % 2] for i in range(n)], dtype=np.int64)
            labels = np.array([str(s) for s in spacegroups])
            label_ids = np.array([i % 2 for i in range(n)])
            role = "spacegroup"
        (stack_dir / "classes.yaml").write_text(
            yaml.safe_dump({"role": role, "classes": classes})
        )
        split = np.array(["train"] * (n - 2) + ["val"] * 2)
        payload = dict(
            embeddings=rng.normal(size=(n, latent_dim)).astype(np.float32),
            labels=labels,
            label_ids=label_ids,
            spacegroups=spacegroups,
            spacegroup_ids=label_ids,
            material_ids=np.array([f"{stack}-{i}" for i in range(n)]),
            split=split,
            feature_mean=np.zeros(3, dtype=np.float32),
            feature_std=np.ones(3, dtype=np.float32),
        )
        if n_features is not None:
            payload["features"] = rng.normal(size=(n, n_features)).astype(np.float32)
        np.savez(stack_dir / "embeddings.npz", **payload)

        for heads_name in heads:
            heads_dir = stack_dir / "heads" / heads_name
            heads_dir.mkdir(parents=True)
            (heads_dir / "heads.yaml").write_text("{}\n")
            probs = np.eye(len(classes), dtype=np.float32)[label_ids]
            np.savez(
                heads_dir / "predictions.npz",
                material_ids=payload["material_ids"],
                split=split,
                label_ids=label_ids,
                probs=probs,
            )
            np.savez(
                heads_dir / "viz_embeddings.npz",
                embeddings=rng.normal(size=(n, 2)).astype(np.float32),
                label_ids=label_ids,
                material_ids=payload["material_ids"],
                split=split,
            )
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "name": run_dir.name,
                "seed": 0,
                "model_kind": model_kind,
                "output_dir": str(run_dir.parent),
                "stacks": specs,
            }
        )
    )
    return run_dir
