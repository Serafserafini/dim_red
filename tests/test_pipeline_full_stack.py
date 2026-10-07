from pathlib import Path

import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.io import write

pytest.importorskip("jax")

from dim_red.pipeline.full_stack import FullStack
from dim_red.pipeline.full_stack_config import FAMILY, full_stack_config_from_dict
from dim_red.pipeline.stack_data import StackDataset

SYSTEM_SPACEGROUPS = {
    "cubic": [195, 196],
    "tetragonal": [75, 76],
    "hexagonal": [168, 169],
}


def _block(**over):
    block = {
        "data": {
            "pyxtal": {"structures_per_spacegroup": 2},
            "soap": {"element_agnostic": True},
        },
        "encoder": {"encoder_hidden_dim": [8], "latent_dim": 4},
        "projection": {"projection_dim": 5},
        "train": {"epochs": 2, "batch_size": 8},
        "classifier": {"hidden_dim": 6},
        "viz": {"hidden_dim": [5]},
        "min_train_rows": 5,
    }
    block.update(over)
    return block


def _config(tmp_path, experts=("cubic", "tetragonal"), **over):
    d = {
        "name": "fs",
        "seed": 3,
        "output_dir": str(tmp_path / "runs"),
        "family": _block(),
        "experts": {"defaults": _block(), **{e: {} for e in experts}},
    }
    d.update(over)
    return full_stack_config_from_dict(d)


def _fake_builder(tmp_path, calls, rows_per_sg=12):
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
        for k, (family, sg) in enumerate(plan):
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


@pytest.fixture
def patched(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "dim_red.pipeline.full_stack.build_stack_dataset",
        _fake_builder(tmp_path, calls),
    )
    return calls


def test_fit_body_trains_every_configured_stack_with_its_own_dataset(tmp_path, patched):
    fs = FullStack.create(_config(tmp_path), cache_dir=tmp_path / "cache")
    histories = fs.fit_body()

    assert set(histories) == {"family", "cubic", "tetragonal"}
    assert [spec.name for spec in patched] == [
        "family",
        "tetragonal",
        "cubic",
    ]  # STACK_ORDER
    assert len({spec.seed for spec in patched}) == 3  # independent seeds

    for name in histories:
        stack_dir = fs.run_dir / "stacks" / name
        for rel in (
            "config.yaml",
            "dataset.extxyz",
            "classes.yaml",
            "embeddings.npz",
            "body/encoder_params.msgpack",
            "body/projection_params.msgpack",
            "body/stack.yaml",
            "body/loss_history.csv",
        ):
            assert (stack_dir / rel).exists(), f"{name}/{rel}"
    with open(fs.run_dir / "stacks" / "cubic" / "classes.yaml") as f:
        classes = yaml.safe_load(f)
    assert classes == {"role": "spacegroup", "classes": [195, 196]}
    with open(fs.run_dir / "stacks" / "family" / "classes.yaml") as f:
        assert yaml.safe_load(f)["role"] == "family"


def test_subset_of_experts_only_creates_those_stacks(tmp_path, patched):
    fs = FullStack.create(_config(tmp_path), cache_dir=tmp_path / "cache")
    fs.fit_body(stacks=["cubic"])
    assert [p.name for p in sorted((fs.run_dir / "stacks").iterdir())] == ["cubic"]


def test_unknown_stack_in_selection_is_rejected(tmp_path, patched):
    fs = FullStack.create(_config(tmp_path), cache_dir=tmp_path / "cache")
    with pytest.raises(ValueError, match="hexagonal"):
        fs.fit_body(stacks=["hexagonal"])
    assert not (fs.run_dir / "stacks").exists()


def test_config_without_family_trains_experts_only(tmp_path, patched):
    d = {
        "name": "fs",
        "output_dir": str(tmp_path / "runs"),
        "experts": {"defaults": _block(), "cubic": {}},
    }
    fs = FullStack.create(full_stack_config_from_dict(d), cache_dir=tmp_path / "cache")
    fs.fit_body()
    assert [p.name for p in (fs.run_dir / "stacks").iterdir()] == ["cubic"]


def test_fit_body_twice_on_the_same_stack_raises(tmp_path, patched):
    fs = FullStack.create(_config(tmp_path), cache_dir=tmp_path / "cache")
    fs.fit_body(stacks=["cubic"])
    with pytest.raises(FileExistsError, match="cubic"):
        fs.fit_body(stacks=["cubic"])


def test_a_stack_can_be_added_later_through_open(tmp_path, patched):
    config = _config(tmp_path, experts=("cubic", "tetragonal"))
    fs = FullStack.create(config, cache_dir=tmp_path / "cache")
    fs.fit_body(stacks=["cubic"])

    reopened = FullStack.open(fs.run_dir, cache_dir=tmp_path / "cache")
    reopened.fit_body(stacks=["tetragonal"])
    assert {p.name for p in (fs.run_dir / "stacks").iterdir()} == {
        "cubic",
        "tetragonal",
    }
    again = FullStack.open(fs.run_dir)
    assert {"cubic", "tetragonal"} <= set(again.config.stacks)


def test_too_few_training_rows_names_the_stack_and_writes_nothing(tmp_path, patched):
    d_block = _block(min_train_rows=10_000)
    d = {
        "output_dir": str(tmp_path / "runs"),
        "experts": {"defaults": d_block, "cubic": {}},
    }
    fs = FullStack.create(full_stack_config_from_dict(d), cache_dir=tmp_path / "cache")
    with pytest.raises(ValueError, match="cubic"):
        fs.fit_body()
    assert not (fs.run_dir / "stacks" / "cubic").exists()


def test_single_class_dataset_is_rejected(tmp_path, monkeypatch):
    calls = []
    builder = _fake_builder(tmp_path, calls)

    def one_class(spec, model_kind, cache_dir):
        ds = builder(spec, model_kind, cache_dir)
        return StackDataset(
            **{**ds.__dict__, "spacegroups": [195] * len(ds.spacegroups)}
        )

    monkeypatch.setattr("dim_red.pipeline.full_stack.build_stack_dataset", one_class)
    d = {
        "output_dir": str(tmp_path / "runs"),
        "experts": {"defaults": _block(), "cubic": {}},
    }
    fs = FullStack.create(full_stack_config_from_dict(d), cache_dir=tmp_path / "cache")
    with pytest.raises(ValueError, match="at least 2") as exc:
        fs.fit_body()
    assert "cubic" in str(exc.value)
    assert not (fs.run_dir / "stacks" / "cubic").exists()


def test_fit_heads_writes_heads_and_does_not_touch_the_body(tmp_path, patched):
    fs = FullStack.create(_config(tmp_path), cache_dir=tmp_path / "cache")
    fs.fit_body()
    body_file = fs.run_dir / "stacks" / "cubic" / "body" / "encoder_params.msgpack"
    before = body_file.read_bytes()

    result = fs.fit_heads("h1")
    assert set(result) == {"family", "cubic", "tetragonal"}
    for name in result:
        heads_dir = fs.run_dir / "stacks" / name / "heads" / "h1"
        for rel in (
            "classifier_params.msgpack",
            "viz_params.msgpack",
            "heads.yaml",
            "classifier_loss_history.csv",
            "viz_loss_history.csv",
            "predictions.npz",
            "viz_embeddings.npz",
            "viz_plot.png",
        ):
            assert (heads_dir / rel).exists(), f"{name}/{rel}"
        preds = np.load(heads_dir / "predictions.npz")
        np.testing.assert_allclose(preds["probs"].sum(axis=1), 1.0, atol=1e-5)
    assert body_file.read_bytes() == before


def test_fit_heads_twice_with_different_names_on_one_body(tmp_path, patched):
    fs = FullStack.create(_config(tmp_path), cache_dir=tmp_path / "cache")
    fs.fit_body(stacks=["cubic"])
    fs.fit_heads("a", stacks=["cubic"])
    other = _config(tmp_path)  # same shape of config, passed explicitly
    fs.fit_heads("b", stacks=["cubic"], config=other)
    heads = {p.name for p in (fs.run_dir / "stacks" / "cubic" / "heads").iterdir()}
    assert heads == {"a", "b"}
    with pytest.raises(FileExistsError, match="a"):
        fs.fit_heads("a", stacks=["cubic"])


def test_fit_heads_requires_a_trained_body(tmp_path, patched):
    fs = FullStack.create(_config(tmp_path), cache_dir=tmp_path / "cache")
    with pytest.raises(FileNotFoundError, match="cubic"):
        fs.fit_heads("h", stacks=["cubic"])


def test_fit_heads_viz_dim_3(tmp_path, patched):
    d = {
        "output_dir": str(tmp_path / "runs"),
        "experts": {
            "defaults": _block(viz={"hidden_dim": [5], "viz_dim": 3}),
            "cubic": {},
        },
    }
    fs = FullStack.create(full_stack_config_from_dict(d), cache_dir=tmp_path / "cache")
    fs.fit_body()
    fs.fit_heads("h")
    heads_dir = fs.run_dir / "stacks" / "cubic" / "heads" / "h"
    assert np.load(heads_dir / "viz_embeddings.npz")["embeddings"].shape[1] == 3
    assert not (heads_dir / "viz_plot.png").exists()


def test_load_stack_returns_a_usable_single_stack(tmp_path, patched):
    fs = FullStack.create(_config(tmp_path), cache_dir=tmp_path / "cache")
    fs.fit_body(stacks=["cubic"])
    fs.fit_heads("h", stacks=["cubic"])
    stack = fs.load_stack("cubic", heads_name="h")
    X = np.load(fs.run_dir / "stacks" / "cubic" / "embeddings.npz")["features"]
    assert stack.predict_proba(X).shape == (len(X), 2)
    assert stack.encode(X).shape == (len(X), 4)


def _fail_once(monkeypatch, target, attr):
    real = getattr(target, attr)
    state = {"n": 0}

    def flaky(*a, **k):
        if state["n"] == 0:
            state["n"] += 1
            raise OSError("disk full")
        return real(*a, **k)

    monkeypatch.setattr(target, attr, flaky)
    return lambda: monkeypatch.setattr(target, attr, real)


def test_failed_body_write_leaves_nothing_and_can_be_retried(
    tmp_path, patched, monkeypatch
):
    fs = FullStack.create(_config(tmp_path), cache_dir=tmp_path / "cache")
    undo = _fail_once(monkeypatch, np, "savez")
    with pytest.raises(OSError, match="disk full"):
        fs.fit_body(stacks=["cubic"])
    undo()
    assert not (fs.run_dir / "stacks" / "cubic").exists()
    assert not (fs.run_dir / "stacks" / ".cubic.tmp").exists()
    fs.fit_body(stacks=["cubic"])
    assert (fs.run_dir / "stacks" / "cubic" / "body" / "stack.yaml").exists()


def test_failed_heads_write_leaves_nothing_and_can_be_retried(
    tmp_path, patched, monkeypatch
):
    fs = FullStack.create(_config(tmp_path), cache_dir=tmp_path / "cache")
    fs.fit_body(stacks=["cubic"])
    undo = _fail_once(monkeypatch, np, "savez")
    with pytest.raises(OSError, match="disk full"):
        fs.fit_heads("h", stacks=["cubic"])
    undo()
    heads = fs.run_dir / "stacks" / "cubic" / "heads"
    assert not (heads / "h").exists()
    assert not (heads / ".h.tmp").exists()
    fs.fit_heads("h", stacks=["cubic"])
    assert (heads / "h" / "heads.yaml").exists()


def _config_b(tmp_path, model_kind="supcon", **block_over):
    block = _block(encoder={"encoder_hidden_dim": [8], "latent_dim": 7})
    block.update(block_over)
    return full_stack_config_from_dict(
        {
            "name": "other",
            "seed": 99,
            "model_kind": model_kind,
            "output_dir": str(tmp_path / "elsewhere"),
            "experts": {"defaults": block, "cubic": {}, "tetragonal": {}},
        }
    )


def _recorded(run_dir):
    from dim_red.pipeline.full_stack_config import full_stack_config_from_resolved_dict

    with open(run_dir / "config.yaml") as f:
        return full_stack_config_from_resolved_dict(yaml.safe_load(f))


def test_open_with_config_uses_it_as_working_config_and_records_new_stacks(
    tmp_path, patched
):
    fs = FullStack.create(
        _config(tmp_path, experts=("cubic",)), cache_dir=tmp_path / "cache"
    )
    fs.fit_body(stacks=["cubic"])

    config_b = _config_b(tmp_path)
    reopened = FullStack.open(fs.run_dir, config=config_b)

    assert reopened.config is config_b
    assert reopened.config.stacks["cubic"].model.latent_dim == 7
    recorded = _recorded(fs.run_dir)
    assert recorded.name == "fs" and recorded.seed == 3
    assert recorded.stacks["cubic"].model.latent_dim == 4  # trained spec kept
    assert recorded.stacks["tetragonal"].model.latent_dim == 7  # new stack recorded

    again = FullStack.open(fs.run_dir)
    assert {"cubic", "tetragonal"} <= set(again.config.stacks)


def test_fit_body_after_open_with_config_selects_the_working_configs_stacks(
    tmp_path, patched
):
    fs = FullStack.create(
        _config(tmp_path, experts=("cubic",)), cache_dir=tmp_path / "cache"
    )
    fs.fit_body(stacks=["cubic"])
    reopened = FullStack.open(
        fs.run_dir, config=_config_b(tmp_path), cache_dir=tmp_path / "cache"
    )
    with pytest.raises(FileExistsError, match="cubic"):
        reopened.fit_body()
    assert not (fs.run_dir / "stacks" / "tetragonal").exists()  # pre-checks first


def test_fit_heads_after_open_with_config_uses_the_working_configs_heads(
    tmp_path, patched
):
    fs = FullStack.create(
        _config(tmp_path, experts=("cubic",)), cache_dir=tmp_path / "cache"
    )
    fs.fit_body(stacks=["cubic"])
    config_b = _config_b(
        tmp_path,
        encoder={"encoder_hidden_dim": [8], "latent_dim": 4},
        classifier={"hidden_dim": 9},
    )
    reopened = FullStack.open(fs.run_dir, config=config_b, cache_dir=tmp_path / "cache")
    reopened.fit_heads("h", stacks=["cubic"])
    with open(fs.run_dir / "stacks" / "cubic" / "heads" / "h" / "heads.yaml") as f:
        assert yaml.safe_load(f)["config"]["classifier_hidden_dim"] == 9


def test_open_with_a_different_model_kind_is_rejected(tmp_path, patched):
    fs = FullStack.create(
        _config(tmp_path, experts=("cubic",)), cache_dir=tmp_path / "cache"
    )
    data = {
        "pyxtal": {"structures_per_spacegroup": 2},
        "mace": {"checkpoint_path": "x.msgpack"},
    }
    config_mace = _config_b(tmp_path, model_kind="supcon_mace", data=data)
    before = (fs.run_dir / "config.yaml").read_bytes()
    with pytest.raises(ValueError, match="supcon_mace") as exc:
        FullStack.open(fs.run_dir, config=config_mace)
    assert "supcon" in str(exc.value)
    assert (fs.run_dir / "config.yaml").read_bytes() == before


def test_open_without_config_is_read_only(tmp_path, patched):
    import os

    fs = FullStack.create(
        _config(tmp_path, experts=("cubic",)), cache_dir=tmp_path / "cache"
    )
    path = fs.run_dir / "config.yaml"
    before = path.read_bytes()
    FullStack.open(fs.run_dir)
    assert path.read_bytes() == before

    if os.geteuid() != 0:
        fs.run_dir.chmod(0o555)
        try:
            FullStack.open(fs.run_dir)
        finally:
            fs.run_dir.chmod(0o755)
    assert path.read_bytes() == before


def test_run_config_write_is_atomic(tmp_path, patched, monkeypatch):
    fs = FullStack.create(
        _config(tmp_path, experts=("cubic",)), cache_dir=tmp_path / "cache"
    )
    path = fs.run_dir / "config.yaml"
    before = path.read_bytes()
    listing = sorted(p.name for p in fs.run_dir.iterdir())

    def boom(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr("dim_red.pipeline.full_stack.os.replace", boom)
    with pytest.raises(OSError, match="replace failed"):
        FullStack.open(fs.run_dir, config=_config_b(tmp_path))
    assert path.read_bytes() == before
    assert sorted(p.name for p in fs.run_dir.iterdir()) == listing


def _boom_for(stack_name, real):
    def build(spec, model_kind, cache_dir):
        if spec.name == stack_name:
            raise RuntimeError("boom")
        return real(spec, model_kind, cache_dir)

    return build


def _assert_names_stack(exc_info, caplog, stack_name, phase):
    assert any(
        stack_name in r.getMessage() and phase in r.getMessage()
        for r in caplog.records
        if r.levelname == "ERROR"
    )
    assert any(stack_name in n for n in getattr(exc_info.value, "__notes__", []))


def test_failure_inside_fit_body_names_the_stack(tmp_path, monkeypatch, caplog):
    real = _fake_builder(tmp_path, [])
    monkeypatch.setattr(
        "dim_red.pipeline.full_stack.build_stack_dataset", _boom_for("cubic", real)
    )
    fs = FullStack.create(_config(tmp_path), cache_dir=tmp_path / "cache")
    with caplog.at_level("ERROR", logger="dim_red.pipeline"):
        with pytest.raises(RuntimeError, match="boom") as exc:
            fs.fit_body()
    _assert_names_stack(exc, caplog, "cubic", "fit_body")


def test_failure_inside_fit_heads_names_the_stack(
    tmp_path, patched, monkeypatch, caplog
):
    fs = FullStack.create(_config(tmp_path), cache_dir=tmp_path / "cache")
    fs.fit_body(stacks=["cubic"])

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(FullStack, "_write_heads", staticmethod(boom))
    with caplog.at_level("ERROR", logger="dim_red.pipeline"):
        with pytest.raises(RuntimeError, match="boom") as exc:
            fs.fit_heads("h", stacks=["cubic"])
    _assert_names_stack(exc, caplog, "cubic", "fit_heads")
