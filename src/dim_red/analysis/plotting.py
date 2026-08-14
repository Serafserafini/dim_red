"""
Plotting utilities for visualization of reduced dimensional spaces.
"""

from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the "3d" projection
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def plot_reduced_space(
    X_reduced: np.ndarray,
    labels: List[str],
    title: str = "PCA Projection of SOAP descriptors",
    save_path: Optional[str] = None,
    xlabel: str = "Principal Component 1",
    ylabel: str = "Principal Component 2",
) -> None:
    """Plots the 2D reduced dimensional space using Matplotlib.

    Args:
        X_reduced: Reduced coordinates of shape (n_samples, 2).
        labels: Class labels (e.g. crystal system names) for color mapping.
        title: The plot title.
        save_path: If provided, saves the plot to this filepath; otherwise the figure is shown (interactively, or captured inline in a notebook).
        xlabel: Label for the x-axis (first reduced dimension).
        ylabel: Label for the y-axis (second reduced dimension).
    """
    if X_reduced.shape[1] < 2:
        raise ValueError("X_reduced must have at least 2 components for a 2D plot.")

    plt.figure(figsize=(10, 8))

    unique_labels = sorted(list(set(labels)))
    cmap = plt.get_cmap("tab10")

    for i, label in enumerate(unique_labels):
        mask = [lbl == label for lbl in labels]
        coords = X_reduced[mask]
        plt.scatter(
            coords[:, 0],
            coords[:, 1],
            label=label,
            color=cmap(i % 10),
            alpha=0.8,
            edgecolors="w",
            s=80,
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved successfully to {save_path}")
    else:
        # No file requested -- assume interactive/notebook use, where the
        # figure must still be open when the inline backend captures it.
        plt.show()

    plt.close()


def plot_reduced_space_3d(
    X_reduced: np.ndarray,
    labels: List[str],
    title: str = "3D Projection",
    save_path: Optional[str] = None,
    xlabel: str = "Dimension 1",
    ylabel: str = "Dimension 2",
    zlabel: str = "Dimension 3",
) -> None:
    """Plots a 3D reduced space using Matplotlib.

    Mirrors ``plot_reduced_space``'s structure/styling (same per-label
    coloring, legend, grid) but scatters into a 3D axes -- used for a
    ``dim_red.supcon.tails.VisualizationTail`` trained with ``output_dim ==
    3`` (see ``dim_red.pipeline.tail_training``), where a plain 2D scatter
    would silently drop the third dimension.

    Args:
        X_reduced: Reduced coordinates of shape ``(n_samples, 3)``.
        labels: Class labels (e.g. crystal family names) for color mapping.
        title: The plot title.
        save_path: If provided, saves the plot to this filepath; otherwise the figure is shown (interactively, or captured inline in a notebook).
        xlabel: Label for the x-axis (first reduced dimension).
        ylabel: Label for the y-axis (second reduced dimension).
        zlabel: Label for the z-axis (third reduced dimension).

    Raises:
        ValueError: If ``X_reduced`` doesn't have exactly 3 components.
    """
    if X_reduced.shape[1] != 3:
        raise ValueError("X_reduced must have exactly 3 components for a 3D plot.")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")

    unique_labels = sorted(list(set(labels)))
    cmap = plt.get_cmap("tab10")

    for i, label in enumerate(unique_labels):
        mask = [lbl == label for lbl in labels]
        coords = X_reduced[mask]
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            label=label,
            color=cmap(i % 10),
            alpha=0.8,
            edgecolors="w",
            s=80,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.legend(frameon=True, facecolor="white", edgecolor="none")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved successfully to {save_path}")
    else:
        # No file requested -- assume interactive/notebook use, where the
        # figure must still be open when the inline backend captures it.
        plt.show()

    plt.close()


def plot_applied_structures(
    X_original: np.ndarray,
    original_labels: List[str],
    X_new: np.ndarray,
    new_labels: Optional[List[str]] = None,
    title: str = "Applied structures in latent space",
    save_path: Optional[str] = None,
    xlabel: str = "Latent Dimension 1",
    ylabel: str = "Latent Dimension 2",
) -> None:
    """Scatter an original dataset's 2D coordinates (colored by label, same
    style as ``plot_reduced_space``) with a second set of points overlaid as
    larger, black-edged stars -- e.g. new structures encoded with an
    already-trained model (see ``dim_red.pipeline.inference``), so they can
    be visually compared against where the training data itself landed.

    Both point sets must already be 2D; projecting a non-2D latent space
    (e.g. via UMAP) is the caller's responsibility, same division of concerns
    as ``dim_red.pipeline.compare``'s own use of ``plot_reduced_space``.

    Args:
        X_original: Original dataset's 2D coordinates, shape ``(n, 2)``.
        original_labels: Label per original point, same length as ``X_original``.
        X_new: New points' 2D coordinates, shape ``(m, 2)``.
        new_labels: Label per new point, same length as ``X_new``. Sharing a
            label with an ``original_labels`` entry reuses that label's
            color (just drawn as a star instead of a circle); a label not
            seen in ``original_labels`` gets its own new color. ``None``
            (default) groups every new point under a single "Applied
            structure" legend entry.
        title: The plot title.
        save_path: If provided, saves the plot to this filepath; otherwise the figure is shown (interactively, or captured inline in a notebook).
        xlabel: Label for the x-axis (first coordinate).
        ylabel: Label for the y-axis (second coordinate).

    Raises:
        ValueError: If ``X_original``/``X_new`` aren't 2D coordinates, ``X_new``
            is empty, or ``new_labels`` doesn't match ``X_new``'s length.
    """
    if X_original.shape[1] < 2 or X_new.shape[1] < 2:
        raise ValueError("X_original and X_new must have at least 2 components.")
    if X_new.shape[0] == 0:
        raise ValueError("X_new must contain at least one point.")
    if new_labels is None:
        new_labels = ["Applied structure"] * X_new.shape[0]
    elif len(new_labels) != X_new.shape[0]:
        raise ValueError("new_labels must have the same length as X_new.")

    unique_labels = sorted(set(original_labels) | set(new_labels))
    cmap = plt.get_cmap("tab10")
    color_for = {label: cmap(i % 10) for i, label in enumerate(unique_labels)}

    plt.figure(figsize=(10, 8))

    for label in sorted(set(original_labels)):
        mask = [lbl == label for lbl in original_labels]
        coords = X_original[mask]
        plt.scatter(
            coords[:, 0],
            coords[:, 1],
            label=label,
            color=color_for[label],
            alpha=0.5,
            edgecolors="w",
            s=60,
            zorder=2,
        )

    for label in sorted(set(new_labels)):
        mask = [lbl == label for lbl in new_labels]
        coords = X_new[mask]
        plt.scatter(
            coords[:, 0],
            coords[:, 1],
            label=f"{label} (applied)",
            color=color_for[label],
            marker="*",
            s=300,
            edgecolors="black",
            linewidths=1.2,
            zorder=3,
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=8)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved successfully to {save_path}")
    else:
        # No file requested -- assume interactive/notebook use, where the
        # figure must still be open when the inline backend captures it.
        plt.show()

    plt.close()


def plot_spacegroup_histogram(
    spacegroups: Sequence[Optional[int]],
    families: Sequence[str],
    title: str = "Spacegroup distribution",
    save_path: Optional[str] = None,
    xlabel: str = "Spacegroup number",
    ylabel: str = "Count",
) -> None:
    """Bar chart of how many structures fall into each Materials Project
    spacegroup number, one bar per spacegroup present in ``spacegroups``,
    colored by the crystal family (crystal system) it belongs to.

    Meant to run right after fetching -- e.g. pairing
    ``[a.info.get("spacegroup") for a in atoms_list]`` with the
    crystal-system label each ``Atoms`` object was fetched under, the same
    "family" grouping ``dim_red.analysis.workflow.run_pca_reduction`` and the
    ``aux_heads`` family-classification head use elsewhere in this package --
    but works equally from a previously saved dataset's ``spacegroups``/
    ``labels`` arrays (e.g. ``dim_red.pipeline.dataset_cache``'s output).

    Args:
        spacegroups: Spacegroup number (1-230) per structure, same length and
            order as ``families``. ``None`` or a non-positive value (e.g. the
            ``-1`` sentinel this package uses elsewhere for structures MP
            returned no symmetry data for) is grouped into an "Unknown" bar.
        families: Crystal family/system label per structure, same length and
            order as ``spacegroups``.
        title: The plot title.
        save_path: If provided, saves the plot to this filepath; otherwise the figure is shown (interactively, or captured inline in a notebook).
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.

    Raises:
        ValueError: If ``spacegroups`` and ``families`` have different
            lengths, or a spacegroup is associated with more than one family
            (which would make its bar's color ambiguous).
    """
    if len(spacegroups) != len(families):
        raise ValueError("spacegroups and families must have the same length.")

    counts: Dict[str, int] = {}
    sg_family: Dict[str, str] = {}
    for spacegroup, family in zip(spacegroups, families):
        key = (
            "Unknown"
            if spacegroup is None or int(spacegroup) < 1
            else str(int(spacegroup))
        )
        counts[key] = counts.get(key, 0) + 1
        if key in sg_family and sg_family[key] != family:
            raise ValueError(
                f"Spacegroup {key!r} is associated with multiple families "
                f"({sg_family[key]!r} and {family!r}); cannot assign a single "
                "bar color."
            )
        sg_family[key] = family

    def _sort_key(sg_key: str):
        return (sg_key == "Unknown", int(sg_key) if sg_key != "Unknown" else -1)

    sg_sorted = sorted(counts, key=_sort_key)
    unique_families = sorted(set(families))
    cmap = plt.get_cmap("tab10")
    family_color = {family: cmap(i % 10) for i, family in enumerate(unique_families)}

    plt.figure(figsize=(max(10, 0.35 * len(sg_sorted)), 6))
    plt.bar(
        sg_sorted,
        [counts[sg] for sg in sg_sorted],
        color=[family_color[sg_family[sg]] for sg in sg_sorted],
    )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=family_color[family])
        for family in unique_families
    ]
    plt.legend(
        handles,
        unique_families,
        title="Family",
        frameon=True,
        facecolor="white",
        edgecolor="none",
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=90, fontsize=7)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved successfully to {save_path}")
    else:
        # No file requested -- assume interactive/notebook use, where the
        # figure must still be open when the inline backend captures it.
        plt.show()

    plt.close()


_MAX_HEATMAP_INCHES = 20.0
_MAX_BAR_CHART_INCHES = 30.0
_MAX_TICK_LABELS = 40
_MAX_ANNOTATED_CM_CLASSES = 25


def _thin_tick_positions(
    n: int, max_labels: int, must_keep: Sequence[int] = ()
) -> np.ndarray:
    """Evenly-spaced tick positions, thinned to at most ``max_labels`` when
    there are more classes than that -- avoids an axis of overlapping,
    unreadable labels for a high-cardinality vocabulary (e.g. up to 230
    Materials Project spacegroups, vs. a handful of crystal families).
    ``must_keep`` positions (e.g. a trailing "macro avg" bar) are always
    included even if regular-interval thinning would otherwise skip them.
    """
    if n <= max_labels:
        return np.arange(n)
    step = int(np.ceil(n / max_labels))
    positions = set(range(0, n, step))
    positions.update(must_keep)
    return np.array(sorted(positions))


def _predicted_labels(y_probs: np.ndarray, class_names: Sequence) -> np.ndarray:
    """Argmax-decode a softmax probability matrix into predicted labels,
    shared by every classifier-evaluation plot below.
    """
    class_names_arr = np.asarray(class_names)
    return class_names_arr[np.argmax(y_probs, axis=1)]


def _validate_probs_shape(
    y_true: Sequence, y_probs: np.ndarray, class_names: Sequence
) -> None:
    if len(y_true) != y_probs.shape[0]:
        raise ValueError(
            "y_true and y_probs must have the same number of rows "
            f"(got {len(y_true)} and {y_probs.shape[0]})."
        )
    if y_probs.shape[1] != len(class_names):
        raise ValueError(
            "y_probs.shape[1] must match len(class_names) "
            f"(got {y_probs.shape[1]} and {len(class_names)})."
        )


def plot_confusion_matrix(
    y_true: Sequence,
    y_probs: np.ndarray,
    class_names: Sequence,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    normalize: bool = True,
) -> None:
    """Plots a confusion matrix heatmap for a classifier's predictions.

    Color encodes a magnitude (count or per-row fraction), not class
    identity, so this uses a single sequential colormap ("Blues") rather
    than the categorical ``tab10`` palette the rest of this module uses for
    per-label coloring.

    Args:
        y_true: True label per point, length ``n``, same value type as
            ``class_names``'s entries (e.g. family name strings, or
            spacegroup integers).
        y_probs: Softmax probabilities, shape ``(n, len(class_names))``.
            The predicted label per point is ``class_names[y_probs.argmax(axis=1)]``.
        class_names: The full class vocabulary, in the order ``y_probs``'s
            columns correspond to.
        title: The plot title (an overall accuracy is appended to it).
        save_path: If provided, saves the plot to this filepath; otherwise the figure is shown (interactively, or captured inline in a notebook).
        normalize: If True (default), each row is normalized to sum to 1 (so
            cells read as per-true-class recall); otherwise raw counts.

    For a high-cardinality ``class_names`` (e.g. up to 230 spacegroups), tick
    labels are thinned to at most ``_MAX_TICK_LABELS`` and the figure size is
    capped at ``_MAX_HEATMAP_INCHES`` rather than growing with ``n`` without
    bound; per-cell count/fraction text is dropped above
    ``_MAX_ANNOTATED_CM_CLASSES`` classes, since it would be both illegible
    and slow to render -- the colorbar-mapped color still conveys the same
    information.

    Raises:
        ValueError: If ``y_true``/``y_probs`` have mismatched lengths, or
            ``y_probs.shape[1]`` doesn't match ``len(class_names)``.
    """
    _validate_probs_shape(y_true, y_probs, class_names)

    y_true_arr = np.asarray(y_true)
    y_pred = _predicted_labels(y_probs, class_names)
    cm = confusion_matrix(y_true_arr, y_pred, labels=list(class_names))
    accuracy = accuracy_score(y_true_arr, y_pred)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_display = np.divide(
            cm, row_sums, out=np.zeros(cm.shape, dtype=float), where=row_sums != 0
        )
    else:
        cm_display = cm.astype(float)

    n = len(class_names)
    tick_labels = [str(c) for c in class_names]
    # Thin tick labels and cap figure growth for a high-cardinality
    # vocabulary (e.g. spacegroups) -- an unbounded 0.6*n figure would
    # otherwise grow to tens of thousands of pixels and every label would
    # still overlap illegibly.
    tick_positions = _thin_tick_positions(n, _MAX_TICK_LABELS)
    rotation = 45 if len(tick_positions) > 6 else 0
    tick_fontsize = 6 if n > _MAX_TICK_LABELS else 10

    fig_size = min(_MAX_HEATMAP_INCHES, max(6.0, 0.6 * n))
    # Extra width beyond the square plot area for the colorbar strip, so a
    # long title doesn't visually collide with its top tick label.
    plt.figure(figsize=(fig_size + 1.8, fig_size))
    plt.imshow(cm_display, cmap="Blues", vmin=0.0)
    plt.colorbar(label="Fraction of true class" if normalize else "Count")

    plt.xticks(
        tick_positions,
        [tick_labels[i] for i in tick_positions],
        rotation=rotation,
        ha="right" if rotation else "center",
        fontsize=tick_fontsize,
    )
    plt.yticks(
        tick_positions,
        [tick_labels[i] for i in tick_positions],
        fontsize=tick_fontsize,
    )

    # Per-cell text becomes both illegible and slow to render past a few
    # hundred cells (e.g. 230 spacegroups squared) -- skip it for a
    # high-cardinality vocabulary and rely on the colorbar-mapped color alone.
    if n <= _MAX_ANNOTATED_CM_CLASSES:
        peak = cm_display.max() if cm_display.max() > 0 else 1.0
        for i in range(n):
            for j in range(n):
                value = cm_display[i, j]
                text = f"{value:.2f}" if normalize else str(int(cm[i, j]))
                plt.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value > peak / 2 else "black",
                )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(
        f"{title} (accuracy={accuracy:.3f})", fontsize=14, fontweight="bold", pad=15
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved successfully to {save_path}")
    else:
        # No file requested -- assume interactive/notebook use, where the
        # figure must still be open when the inline backend captures it.
        plt.show()

    plt.close()


def plot_classification_report(
    y_true: Sequence,
    y_probs: np.ndarray,
    class_names: Sequence,
    title: str = "Per-class precision / recall / F1",
    save_path: Optional[str] = None,
) -> None:
    """Grouped bar chart of per-class precision/recall/F1 (plus a trailing
    "macro avg" group), via ``sklearn.metrics.classification_report``.

    Color here encodes which *metric* a bar is (precision/recall/F1-score),
    not class identity -- three fixed ``tab10`` colors, consistently
    assigned and never cycled per class.

    Args:
        y_true: True label per point, length ``n``, same value type as
            ``class_names``'s entries.
        y_probs: Softmax probabilities, shape ``(n, len(class_names))``.
        class_names: The full class vocabulary, in the order ``y_probs``'s
            columns correspond to.
        title: The plot title.
        save_path: If provided, saves the plot to this filepath; otherwise the figure is shown (interactively, or captured inline in a notebook).

    For a high-cardinality ``class_names`` (e.g. up to 230 spacegroups), x
    labels are thinned to at most ``_MAX_TICK_LABELS`` (the trailing "macro
    avg" bar is always kept) and the figure width is capped at
    ``_MAX_BAR_CHART_INCHES`` rather than growing with ``n_groups`` without
    bound.

    Raises:
        ValueError: If ``y_true``/``y_probs`` have mismatched lengths, or
            ``y_probs.shape[1]`` doesn't match ``len(class_names)``.
    """
    _validate_probs_shape(y_true, y_probs, class_names)

    y_true_arr = np.asarray(y_true)
    y_pred = _predicted_labels(y_probs, class_names)
    report = classification_report(
        y_true_arr,
        y_pred,
        labels=list(class_names),
        output_dict=True,
        zero_division=0,
    )

    metrics = ["precision", "recall", "f1-score"]
    cmap = plt.get_cmap("tab10")
    metric_color = {metric: cmap(i) for i, metric in enumerate(metrics)}

    group_labels = [str(c) for c in class_names] + ["macro avg"]
    n_groups = len(group_labels)
    x = np.arange(n_groups)
    width = 0.25

    fig_width = min(_MAX_BAR_CHART_INCHES, max(10.0, 0.6 * n_groups))
    plt.figure(figsize=(fig_width, 6))
    for i, metric in enumerate(metrics):
        values = [report[str(c)][metric] for c in class_names]
        values.append(report["macro avg"][metric])
        plt.bar(
            x + (i - 1) * width,
            values,
            width=width,
            label=metric.replace("-score", "").title(),
            color=metric_color[metric],
        )

    # Thin x labels for a high-cardinality vocabulary (e.g. spacegroups) so
    # they don't overlap into an unreadable axis; the trailing "macro avg"
    # bar is always labeled regardless of thinning.
    tick_positions = _thin_tick_positions(
        n_groups, _MAX_TICK_LABELS, must_keep=(n_groups - 1,)
    )
    rotation = 45 if len(tick_positions) > 6 else 0
    tick_fontsize = 6 if n_groups > _MAX_TICK_LABELS else 10
    plt.xticks(
        tick_positions,
        [group_labels[i] for i in tick_positions],
        rotation=rotation,
        ha="right" if rotation else "center",
        fontsize=tick_fontsize,
    )
    plt.ylim(0.0, 1.05)
    plt.ylabel("Score")
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved successfully to {save_path}")
    else:
        # No file requested -- assume interactive/notebook use, where the
        # figure must still be open when the inline backend captures it.
        plt.show()

    plt.close()


def plot_reliability_diagram(
    y_true: Sequence,
    y_probs: np.ndarray,
    class_names: Sequence,
    title: str = "Calibration (reliability diagram)",
    save_path: Optional[str] = None,
    n_bins: int = 10,
) -> None:
    """Reliability diagram: empirical accuracy vs. predicted confidence,
    binned into ``n_bins`` equal-width confidence bins, against a dashed
    "perfectly calibrated" ``y = x`` reference line. Marker size scales with
    how many points fall into each bin.

    Confidence is the predicted class's own probability
    (``y_probs.max(axis=1)``) -- a single curve regardless of how many
    classes there are, unlike a one-vs-rest per-class breakdown, which would
    get cluttered for a many-class vocabulary (e.g. spacegroups).

    Args:
        y_true: True label per point, length ``n``, same value type as
            ``class_names``'s entries.
        y_probs: Softmax probabilities, shape ``(n, len(class_names))``.
        class_names: The full class vocabulary, in the order ``y_probs``'s
            columns correspond to.
        title: The plot title.
        save_path: If provided, saves the plot to this filepath; otherwise the figure is shown (interactively, or captured inline in a notebook).
        n_bins: Number of equal-width confidence bins in ``[0, 1]``.

    Raises:
        ValueError: If ``n_bins`` isn't a positive integer, ``y_true``/
            ``y_probs`` have mismatched lengths, or ``y_probs.shape[1]``
            doesn't match ``len(class_names)``.
    """
    if n_bins <= 0:
        raise ValueError("n_bins must be a positive integer.")
    _validate_probs_shape(y_true, y_probs, class_names)

    y_true_arr = np.asarray(y_true)
    class_names_arr = np.asarray(class_names)
    pred_idx = np.argmax(y_probs, axis=1)
    confidences = y_probs[np.arange(len(y_true_arr)), pred_idx]
    correct = class_names_arr[pred_idx] == y_true_arr

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(confidences, bin_edges[1:-1]), 0, n_bins - 1)

    bin_confidence, bin_accuracy, bin_count = [], [], []
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        if count == 0:
            continue
        bin_confidence.append(float(confidences[mask].mean()))
        bin_accuracy.append(float(correct[mask].mean()))
        bin_count.append(count)

    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")

    if bin_count:
        sizes = 40.0 + 260.0 * (np.asarray(bin_count) / max(bin_count))
        color = plt.get_cmap("tab10")(0)
        plt.plot(bin_confidence, bin_accuracy, color=color, alpha=0.6, zorder=2)
        plt.scatter(
            bin_confidence,
            bin_accuracy,
            s=sizes,
            color=color,
            edgecolors="w",
            zorder=3,
            label="Observed (size = bin count)",
        )

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Predicted confidence")
    plt.ylabel("Empirical accuracy")
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved successfully to {save_path}")
    else:
        # No file requested -- assume interactive/notebook use, where the
        # figure must still be open when the inline backend captures it.
        plt.show()

    plt.close()
