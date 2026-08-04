"""
Unit tests for the analysis plotting utilities.
"""

import pytest

from dim_red.analysis.plotting import plot_spacegroup_histogram

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
