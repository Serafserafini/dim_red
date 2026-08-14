"""
Analysis sub-package for running dimensionality reduction workflows and plotting.
"""

from dim_red.analysis.metrics import embedding_quality_metrics
from dim_red.analysis.plotting import plot_reduced_space, plot_spacegroup_histogram
from dim_red.analysis.workflow import run_pca_reduction

__all__ = [
    "run_pca_reduction",
    "plot_reduced_space",
    "plot_spacegroup_histogram",
    "embedding_quality_metrics",
]
