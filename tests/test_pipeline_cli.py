"""
Unit tests for the pipeline CLI's --compare/--data-file flag handling.
"""

import argparse
from unittest.mock import patch

import pytest

from dim_red.pipeline.cli import _str_to_bool, compare_command, main
from dim_red.pipeline.compare import LatentUmapParams


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
