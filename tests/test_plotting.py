"""
Unit tests for the analysis plotting utilities.
"""

import numpy as np
import pytest

from dim_red.analysis.plotting import plot_applied_structures, plot_spacegroup_histogram

matplotlib = pytest.importorskip("matplotlib")


def test_plot_spacegroup_histogram_writes_file(tmp_path):
    spacegroups = [225, 225, 194]
    families = ["Cubic", "Cubic", "Hexagonal"]
    out = tmp_path / "spacegroups.png"

    plot_spacegroup_histogram(spacegroups, families, save_path=str(out))

    assert out.exists()


def test_plot_spacegroup_histogram_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        plot_spacegroup_histogram([225], ["Cubic", "Hexagonal"])


def test_plot_spacegroup_histogram_rejects_spacegroup_shared_across_families():
    with pytest.raises(ValueError, match="multiple families"):
        plot_spacegroup_histogram([225, 225], ["Cubic", "Hexagonal"])


def test_plot_spacegroup_histogram_groups_missing_spacegroups_as_unknown(tmp_path):
    spacegroups = [225, None, -1]
    families = ["Cubic", "Cubic", "Cubic"]
    out = tmp_path / "spacegroups.png"

    # Should not raise -- same family for all "Unknown" entries (None and the
    # -1 sentinel both count as missing) -- and produce a plot.
    plot_spacegroup_histogram(spacegroups, families, save_path=str(out))
    assert out.exists()


def test_plot_applied_structures_writes_file(tmp_path):
    X_original = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]])
    original_labels = ["Cubic", "Cubic", "Hexagonal"]
    X_new = np.array([[0.1, 0.1]])
    out = tmp_path / "applied.png"

    plot_applied_structures(X_original, original_labels, X_new, save_path=str(out))

    assert out.exists()


def test_plot_applied_structures_with_new_labels(tmp_path):
    X_original = np.array([[0.0, 0.0], [1.0, 1.0]])
    original_labels = ["Cubic", "Hexagonal"]
    X_new = np.array([[0.1, 0.1], [5.0, 5.0]])
    new_labels = ["Cubic", "Unknown"]
    out = tmp_path / "applied.png"

    plot_applied_structures(
        X_original, original_labels, X_new, new_labels=new_labels, save_path=str(out)
    )

    assert out.exists()


def test_plot_applied_structures_rejects_empty_new_points():
    X_original = np.array([[0.0, 0.0]])
    with pytest.raises(ValueError, match="at least one point"):
        plot_applied_structures(X_original, ["Cubic"], np.empty((0, 2)))


def test_plot_applied_structures_rejects_mismatched_new_labels():
    X_original = np.array([[0.0, 0.0]])
    X_new = np.array([[0.1, 0.1], [0.2, 0.2]])
    with pytest.raises(ValueError, match="same length as X_new"):
        plot_applied_structures(X_original, ["Cubic"], X_new, new_labels=["only-one"])
