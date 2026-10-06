"""
Analysis sub-package: embedding-quality metrics and plotting.
"""

from dim_red.analysis.metrics import embedding_quality_metrics
from dim_red.analysis.plotting import plot_reduced_space, plot_spacegroup_histogram

__all__ = [
    "plot_reduced_space",
    "plot_spacegroup_histogram",
    "embedding_quality_metrics",
]
