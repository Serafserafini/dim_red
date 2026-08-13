"""
Unit tests for the analysis plotting utilities.
"""

import numpy as np
import pytest

from dim_red.analysis.plotting import (
    plot_applied_structures,
    plot_classification_report,
    plot_confusion_matrix,
    plot_reduced_space_3d,
    plot_reliability_diagram,
    plot_spacegroup_histogram,
)

matplotlib = pytest.importorskip("matplotlib")


def test_plot_reduced_space_3d_writes_file(tmp_path):
    X = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [5.0, 5.0, 5.0]])
    labels = ["Cubic", "Cubic", "Hexagonal"]
    out = tmp_path / "reduced3d.png"

    plot_reduced_space_3d(X, labels, save_path=str(out))

    assert out.exists()


def test_plot_reduced_space_3d_rejects_non_3d_input():
    X_2d = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="exactly 3 components"):
        plot_reduced_space_3d(X_2d, ["Cubic", "Hexagonal"])

    X_4d = np.array([[0.0, 0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="exactly 3 components"):
        plot_reduced_space_3d(X_4d, ["Cubic"])


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


def _fake_classifier_predictions(n=60, seed=0):
    """A synthetic (y_true, y_probs, class_names) triple for a 3-class
    classifier that's mostly-but-not-perfectly accurate -- used across the
    classifier-evaluation plot tests below.
    """
    rng = np.random.default_rng(seed)
    class_names = ["cubic", "hexagonal", "orthorhombic"]
    y_true = rng.choice(class_names, size=n)
    logits = rng.normal(size=(n, len(class_names)))
    for i, label in enumerate(y_true):
        logits[i, class_names.index(label)] += 2.0
    y_probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return y_true, y_probs, class_names


def test_plot_confusion_matrix_writes_file(tmp_path):
    y_true, y_probs, class_names = _fake_classifier_predictions()
    out = tmp_path / "confusion_matrix.png"

    plot_confusion_matrix(y_true, y_probs, class_names, save_path=str(out))

    assert out.exists()


def test_plot_confusion_matrix_supports_raw_counts_and_integer_classes(tmp_path):
    spacegroup_classes = [1, 2, 5, 12]
    rng = np.random.default_rng(1)
    y_true = rng.choice(spacegroup_classes, size=40)
    logits = rng.normal(size=(40, len(spacegroup_classes)))
    y_probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    out = tmp_path / "confusion_matrix_sg.png"

    plot_confusion_matrix(
        y_true, y_probs, spacegroup_classes, save_path=str(out), normalize=False
    )

    assert out.exists()


def test_plot_confusion_matrix_rejects_mismatched_row_count():
    y_true, y_probs, class_names = _fake_classifier_predictions()
    with pytest.raises(ValueError, match="same number of rows"):
        plot_confusion_matrix(y_true[:5], y_probs, class_names)


def test_plot_confusion_matrix_rejects_mismatched_class_count():
    y_true, y_probs, class_names = _fake_classifier_predictions()
    with pytest.raises(ValueError, match="must match len\\(class_names\\)"):
        plot_confusion_matrix(y_true, y_probs, class_names[:2])


def test_plot_classification_report_writes_file(tmp_path):
    y_true, y_probs, class_names = _fake_classifier_predictions()
    out = tmp_path / "classification_report.png"

    plot_classification_report(y_true, y_probs, class_names, save_path=str(out))

    assert out.exists()


def test_plot_classification_report_rejects_mismatched_row_count():
    y_true, y_probs, class_names = _fake_classifier_predictions()
    with pytest.raises(ValueError, match="same number of rows"):
        plot_classification_report(y_true[:5], y_probs, class_names)


def test_plot_classification_report_rejects_mismatched_class_count():
    y_true, y_probs, class_names = _fake_classifier_predictions()
    with pytest.raises(ValueError, match="must match len\\(class_names\\)"):
        plot_classification_report(y_true, y_probs, class_names[:2])


def test_plot_reliability_diagram_writes_file(tmp_path):
    y_true, y_probs, class_names = _fake_classifier_predictions()
    out = tmp_path / "calibration.png"

    plot_reliability_diagram(y_true, y_probs, class_names, save_path=str(out))

    assert out.exists()


def test_plot_reliability_diagram_rejects_non_positive_n_bins():
    y_true, y_probs, class_names = _fake_classifier_predictions()
    with pytest.raises(ValueError, match="n_bins must be a positive integer"):
        plot_reliability_diagram(y_true, y_probs, class_names, n_bins=0)


def test_plot_reliability_diagram_rejects_mismatched_row_count():
    y_true, y_probs, class_names = _fake_classifier_predictions()
    with pytest.raises(ValueError, match="same number of rows"):
        plot_reliability_diagram(y_true[:5], y_probs, class_names)
