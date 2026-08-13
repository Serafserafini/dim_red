"""
Unit tests for the pipeline CLI's --compare/--data-file flag handling.
"""

import argparse
import dataclasses
from unittest.mock import patch

import pytest

from dim_red.pipeline.cli import (
    _str_to_bool,
    apply_command,
    compare_command,
    main,
    train_tail_command,
)
from dim_red.pipeline.compare import LatentUmapParams
from dim_red.pipeline.config import load_tail_train_config


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
    )


@patch("dim_red.pipeline.compare.generate_comparison_report")
def test_main_compare_flag_defaults_to_no_data_files(mock_generate, tmp_path):
    mock_generate.return_value = tmp_path / "comparison"
    main(["--compare", str(tmp_path)])
    mock_generate.assert_called_once_with(
        str(tmp_path),
        output_dir=None,
        write_data_files=False,
        umap_params=LatentUmapParams(),
    )


@patch("dim_red.pipeline.compare.generate_comparison_report")
def test_main_compare_flag_enables_data_files_with_flag(mock_generate, tmp_path):
    mock_generate.return_value = tmp_path / "comparison"
    main(["--compare", str(tmp_path), "--data-file", "true"])
    mock_generate.assert_called_once_with(
        str(tmp_path),
        output_dir=None,
        write_data_files=True,
        umap_params=LatentUmapParams(),
    )


@patch("dim_red.pipeline.compare.generate_comparison_report")
def test_main_compare_flag_forwards_umap_flags(mock_generate, tmp_path):
    mock_generate.return_value = tmp_path / "comparison"
    main(["--compare", str(tmp_path), "--umap-n-neighbors", "7"])
    mock_generate.assert_called_once_with(
        str(tmp_path),
        output_dir=None,
        write_data_files=False,
        umap_params=LatentUmapParams(n_neighbors=7),
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


@patch("dim_red.pipeline.inference.apply_model_to_structures")
def test_main_apply_flag_forwards_args(mock_apply, tmp_path):
    mock_apply.return_value = tmp_path / "applied"
    main(
        [
            "--apply",
            str(tmp_path / "structures.extxyz"),
            "--apply-run",
            str(tmp_path / "run_dir"),
            "--umap-n-neighbors",
            "5",
        ]
    )
    mock_apply.assert_called_once_with(
        str(tmp_path / "run_dir"),
        str(tmp_path / "structures.extxyz"),
        output_dir=None,
        label_field=None,
        umap_params=LatentUmapParams(n_neighbors=5),
    )


def test_main_apply_flag_requires_apply_run(tmp_path):
    with pytest.raises(SystemExit):
        main(["--apply", str(tmp_path / "structures.extxyz")])


def _write_tail_config(tmp_path, contents):
    path = tmp_path / "tail.yaml"
    path.write_text(contents)
    return path


@patch("dim_red.pipeline.tail_training.train_tail")
def test_train_tail_command_forwards_parsed_config(mock_train_tail, tmp_path):
    config_path = _write_tail_config(
        tmp_path,
        "tail_kind: classification\n" "classification:\n" "  mode: family_only\n",
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
        "  mode: family_only\n",
    )
    run_dir = str(tmp_path / "runs/example")
    mock_train_tail.return_value = tmp_path / "runs/example/tails/classification"

    train_tail_command([str(config_path), run_dir])

    assert mock_train_tail.call_args[0][0].run_dir == run_dir


@patch("dim_red.pipeline.tail_training.train_tail")
def test_main_train_tail_flag_forwards_parsed_config(mock_train_tail, tmp_path):
    config_path = _write_tail_config(
        tmp_path,
        "tail_kind: visualization\n" "visualization:\n" "  viz_dim: 3\n",
    )
    run_dir = str(tmp_path / "runs/example")
    mock_train_tail.return_value = tmp_path / "runs/example/tails/visualization"

    main(["--train-tail", str(config_path), "--train-tail-run", run_dir])

    expected_config = dataclasses.replace(
        load_tail_train_config(config_path), run_dir=run_dir
    )
    mock_train_tail.assert_called_once_with(expected_config)


def test_main_train_tail_flag_requires_train_tail_run(tmp_path):
    config_path = _write_tail_config(
        tmp_path, "tail_kind: classification\nclassification:\n  mode: family_only\n"
    )
    with pytest.raises(SystemExit):
        main(["--train-tail", str(config_path)])
