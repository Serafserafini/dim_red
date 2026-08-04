"""
Plotting utilities for visualization of reduced dimensional spaces.
"""

from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


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
