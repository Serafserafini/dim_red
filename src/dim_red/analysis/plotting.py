"""
Plotting utilities for visualization of reduced dimensional spaces.
"""

from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt


def plot_reduced_space(
    X_reduced: np.ndarray,
    labels: List[str],
    title: str = "PCA Projection of SOAP descriptors",
    save_path: Optional[str] = None
) -> None:
    """Plots the 2D reduced dimensional space using Matplotlib.

    Args:
        X_reduced: Reduced coordinates of shape (n_samples, 2).
        labels: Class labels (e.g. crystal system names) for color mapping.
        title: The plot title.
        save_path: If provided, saves the plot to this filepath.
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
            s=80
        )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved successfully to {save_path}")
    
    plt.close()
