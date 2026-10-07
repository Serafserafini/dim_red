import argparse
import csv
import dataclasses
from unittest.mock import patch

import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.io import write

pytest.importorskip("jax")

from dim_red.pipeline import cli
from dim_red.pipeline import full_stack as fs_mod
from dim_red.pipeline.cli import (
    _str_to_bool,
    apply_command,
    compare_command,
    train_tail_command,
)
from dim_red.pipeline.compare import LatentUmapParams
from dim_red.pipeline.config import load_tail_train_config
from dim_red.pipeline.run_layout import head_names, trained_stack_names
from tests.fullstack_helpers import config_dict, fake_builder


@pytest.fixture
def patched(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "dim_red.pipeline.full_stack.build_stack_dataset", fake_builder(tmp_path, [])
    )


def _write_config(tmp_path, name="cfg.yaml", **kwargs):
    path = tmp_path / name
    path.write_text(
        yaml.safe_dump(config_dict(tmp_path / "runs", experts=("cubic",), **kwargs))
    )
    return str(path)


def test_config_kind_is_detected_by_key():
    assert cli._is_full_stack_config({"model_kind": "supcon"})
    assert cli._is_full_stack_config({"family": {}})
    assert cli._is_full_stack_config({"experts": {}})
    assert not cli._is_full_stack_config({"model": "cgcnn", "encoder": {}})


def test_run_trains_body_and_default_heads(tmp_path, patched):
    run_dir = cli._do_run(_write_config(tmp_path), cache_dir=str(tmp_path / "cache"))
    assert trained_stack_names(run_dir) == ["family", "cubic"]
    assert head_names(run_dir / "stacks" / "family") == ["default"]


def test_run_dispatches_cgcnn_configs_to_run_single(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "dim_red.pipeline.single_run.run_single",
        lambda config, cache_dir=None, **kw: seen.setdefault("kind", config.model_kind)
        and tmp_path,
    )
    path = tmp_path / "c.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model": "cgcnn",
                "data_source": "pyxtal",
                "pyxtal": {"structures_per_spacegroup": 1},
                "encoder": {"encoder_hidden_dim": [8], "latent_dim": 4},
                "aux_heads": {"mode": "family_only"},
            }
        )
    )
    cli._do_run(str(path), None)
    assert seen["kind"] == "cgcnn"


def test_rerun_creates_a_new_run_dir_with_the_same_stacks(tmp_path, patched):
    first = cli._do_run(_write_config(tmp_path), cache_dir=str(tmp_path / "cache"))
    second = cli._do_rerun(str(first), cache_dir=str(tmp_path / "cache"))
    assert second != first and second.parent == first.parent
    assert trained_stack_names(second) == trained_stack_names(first)


def test_train_heads_adds_a_named_heads_set(tmp_path, patched):
    run = cli._do_run(_write_config(tmp_path), cache_dir=str(tmp_path / "cache"))
    heads_cfg = _write_config(tmp_path, name="heads.yaml", latent_dim=4)
    out = cli._do_train_heads(heads_cfg, str(run), heads_name="wide", stacks=["family"])
    assert head_names(run / "stacks" / "family") == ["default", "wide"]
    assert head_names(run / "stacks" / "cubic") == ["default"]
    assert out == run


def test_train_heads_refuses_an_existing_name(tmp_path, patched):
    run = cli._do_run(_write_config(tmp_path), cache_dir=str(tmp_path / "cache"))
    with pytest.raises(FileExistsError):
        cli._do_train_heads(_write_config(tmp_path, name="h.yaml"), str(run), "default")


def test_apply_on_a_fullstack_run_writes_predictions(tmp_path, patched, monkeypatch):
    run = cli._do_run(_write_config(tmp_path), cache_dir=str(tmp_path / "cache"))
    monkeypatch.setattr(
        fs_mod,
        "featurize_structures",
        lambda atoms, kind, data: np.stack(
            [np.arange(6, dtype=np.float32) + i for i in range(len(atoms))]
        ),
    )
    path = tmp_path / "new.extxyz"
    write(str(path), [Atoms("Cu", positions=[[0, 0, 0]])] * 2, format="extxyz")
    out = cli._do_apply(str(path), str(run), None, None)
    assert len(list(csv.DictReader(open(out / "new_predictions.csv")))) == 2


def test_sweep_runs_every_combination(tmp_path, patched):
    sweep = tmp_path / "sweep.yaml"
    sweep.write_text(
        yaml.safe_dump(
            {
                "base": config_dict(tmp_path / "runs", experts=("cubic",)),
                "grid": {"family.encoder.latent_dim": [3, 5]},
            }
        )
    )
    run_dirs = cli._do_sweep(str(sweep), str(tmp_path / "cache"))
    assert len(run_dirs) == 2


def test_compare_and_benchmark_forward_heads_name(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "dim_red.pipeline.compare.generate_comparison_report",
        lambda sweep_dir, **kw: seen.update(compare=kw) or tmp_path,
    )
    monkeypatch.setattr(
        "dim_red.pipeline.benchmark.generate_benchmark_table",
        lambda inputs, out, **kw: seen.update(bench=kw) or tmp_path,
    )
    cli._do_compare("s", None, heads_name="h")
    cli._do_benchmark(["s"], str(tmp_path / "t.csv"), None, heads_name="h")
    assert seen["compare"]["heads_name"] == "h"
    assert seen["bench"]["heads_name"] == "h"


def test_train_heads_console_script_is_registered():
    import tomllib
    from pathlib import Path

    scripts = tomllib.loads(Path("pyproject.toml").read_text())["project"]["scripts"]
    assert scripts["dimred-train-heads"] == "dim_red.pipeline.cli:train_heads_command"


def test_str_to_bool_accepts_common_true_false_spellings():
    for value in ["true", "True", "1", "yes"]:
        assert _str_to_bool(value) is True
    for value in ["false", "False", "0", "no"]:
        assert _str_to_bool(value) is False


def test_str_to_bool_rejects_unknown_value():
    with pytest.raises(argparse.ArgumentTypeError):
        _str_to_bool("maybe")


@patch("dim_red.pipeline.compare.generate_comparison_report")
def test_compare_command_defaults_to_no_data_files(mock_generate, tmp_path):
    mock_generate.return_value = tmp_path / "comparison"
    compare_command([str(tmp_path)])
    mock_generate.assert_called_once_with(
        str(tmp_path),
        output_dir=None,
        write_data_files=False,
        umap_params=LatentUmapParams(),
        heads_name=None,
    )


@patch("dim_red.pipeline.compare.generate_comparison_report")
def test_compare_command_enables_data_files_with_flag(mock_generate, tmp_path):
    mock_generate.return_value = tmp_path / "comparison"
    compare_command([str(tmp_path), "--data-file", "true"])
    mock_generate.assert_called_once_with(
        str(tmp_path),
        output_dir=None,
        write_data_files=True,
        umap_params=LatentUmapParams(),
        heads_name=None,
    )


@patch("dim_red.pipeline.compare.generate_comparison_report")
def test_compare_command_forwards_umap_flags(mock_generate, tmp_path):
    mock_generate.return_value = tmp_path / "comparison"
    compare_command(
        [
            str(tmp_path),
            "--umap-n-neighbors",
            "10",
            "--umap-min-dist",
            "0.2",
            "--umap-metric",
            "cosine",
            "--umap-random-state",
            "3",
        ]
    )
    mock_generate.assert_called_once_with(
        str(tmp_path),
        output_dir=None,
        write_data_files=False,
        umap_params=LatentUmapParams(
            n_neighbors=10, min_dist=0.2, metric="cosine", random_state=3
        ),
        heads_name=None,
    )


@patch("dim_red.pipeline.inference.apply_model_to_structures")
def test_apply_command_forwards_args(mock_apply, tmp_path):
    mock_apply.return_value = tmp_path / "applied"
    apply_command(
        [
            str(tmp_path / "structures.extxyz"),
            str(tmp_path / "run_dir"),
            "--output-dir",
            str(tmp_path / "out"),
            "--label-field",
            "family",
        ]
    )
    mock_apply.assert_called_once_with(
        str(tmp_path / "run_dir"),
        str(tmp_path / "structures.extxyz"),
        output_dir=str(tmp_path / "out"),
        label_field="family",
        umap_params=LatentUmapParams(),
    )


@patch("dim_red.pipeline.inference.apply_model_to_structures")
def test_apply_command_defaults(mock_apply, tmp_path):
    mock_apply.return_value = tmp_path / "applied"
    apply_command([str(tmp_path / "structures.extxyz"), str(tmp_path / "run_dir")])
    mock_apply.assert_called_once_with(
        str(tmp_path / "run_dir"),
        str(tmp_path / "structures.extxyz"),
        output_dir=None,
        label_field=None,
        umap_params=LatentUmapParams(),
    )


def _write_tail_config(tmp_path, contents):
    path = tmp_path / "tail.yaml"
    path.write_text(contents)
    return path


@patch("dim_red.pipeline.tail_training.train_tail")
def test_train_tail_command_forwards_parsed_config(mock_train_tail, tmp_path):
    config_path = _write_tail_config(
        tmp_path,
        "tail_kind: classification\n" "classification:\n" "  head_hidden_dim: 8\n",
    )
    run_dir = str(tmp_path / "runs/example")
    mock_train_tail.return_value = tmp_path / "runs/example/tails/classification"

    train_tail_command([str(config_path), run_dir])

    expected_config = dataclasses.replace(
        load_tail_train_config(config_path), run_dir=run_dir
    )
    mock_train_tail.assert_called_once_with(expected_config)


@patch("dim_red.pipeline.tail_training.train_tail")
def test_train_tail_command_run_dir_overrides_config_run_dir(mock_train_tail, tmp_path):
    """The run_dir positional always wins, even when the YAML also sets one --
    lets the same tail config be reused across many runs without editing it.
    """
    config_path = _write_tail_config(
        tmp_path,
        "run_dir: runs/some-other-run\n"
        "tail_kind: classification\n"
        "classification:\n"
        "  head_hidden_dim: 8\n",
    )
    run_dir = str(tmp_path / "runs/example")
    mock_train_tail.return_value = tmp_path / "runs/example/tails/classification"

    train_tail_command([str(config_path), run_dir])

    assert mock_train_tail.call_args[0][0].run_dir == run_dir
