"""
dim_red - Dimensionality Reduction Package
"""

from dim_red.pca import PCA
from dim_red.utils import standardize

__version__ = "0.1.0"
__all__ = ["PCA", "standardize"]
