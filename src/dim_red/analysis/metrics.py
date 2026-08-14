"""
Model-free, dimensionality-agnostic embedding quality metrics.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

logger = logging.getLogger("dim_red.analysis")


def embedding_quality_metrics(
    embeddings: np.ndarray,
    label_sets: Dict[str, np.ndarray],
    *,
    knn_k: int = 5,
    knn_cv_folds: int = 5,
    kmeans_seed: int = 0,
) -> Dict[str, float]:
    """Dimensionality-agnostic quality scores for a trained embedding, one
    set of scores per named label array in ``label_sets`` (e.g.
    ``{"family": ..., "spacegroup": ...}``) -- computed directly on
    ``embeddings``'s own coordinates, unlike ``dim_red.pipeline.compare``'s
    plotting suite, which projects non-2D embeddings through UMAP first.
    This is what makes one call comparable across every ``model_kind``
    (vae/autoencoder/supcon/cgcnn) regardless of ``latent_dim`` or
    architecture: cgcnn never has a flat ``features`` array and supcon never
    has built-in classifier heads, but every ``model_kind``'s
    ``embeddings.npz`` unconditionally saves ``embeddings``/``labels``/
    ``spacegroups`` (``dim_red.pipeline.single_run``), which is all this
    function needs.

    Rows with a negative label (the sentinel ``dim_red.fetch`` uses for an
    unknown Materials Project spacegroup) are excluded from that label set's
    metrics only, not from any other name's.

    For each name with >= 2 distinct valid labels and enough valid samples
    for the requested folds:
      - ``f"{name}_silhouette"``: ``sklearn.metrics.silhouette_score``, in
        ``[-1, 1]``.
      - ``f"{name}_kmeans_ari"`` / ``f"{name}_kmeans_nmi"``: fit
        ``KMeans(n_clusters=<distinct valid labels>)`` on the embedding,
        score the cluster assignment against the true labels with
        ``adjusted_rand_score``/``normalized_mutual_info_score``.
      - ``f"{name}_knn_accuracy"``: mean cross-validated accuracy of a plain
        k-NN classifier trained directly on the embedding coordinates -- a
        model-free sanity check ("are same-label points actually near each
        other here"), independent of whatever classifier head (if any) a
        given run trained.

    Degenerate inputs (fewer than 2 valid classes, or a class with fewer
    samples than the requested CV folds) return NaN for that name's metrics
    (not omitted, so every call's return dict has the same key set) and log
    a warning rather than raising -- a benchmark table walking many runs
    should not abort on one under-populated label set.
    """
    embeddings = np.asarray(embeddings)
    metrics: Dict[str, float] = {}
    for name, raw_labels in label_sets.items():
        raw_labels = np.asarray(raw_labels)
        valid = raw_labels >= 0
        x = embeddings[valid]
        y = raw_labels[valid]

        keys = (
            f"{name}_silhouette",
            f"{name}_kmeans_ari",
            f"{name}_kmeans_nmi",
            f"{name}_knn_accuracy",
        )
        classes, class_counts = np.unique(y, return_counts=True)
        n_classes = len(classes)
        if n_classes < 2:
            logger.warning(
                "embedding_quality_metrics: label set %r has %d distinct "
                "valid label(s) (need >= 2) -- reporting NaN for it.",
                name,
                n_classes,
            )
            metrics.update({k: float("nan") for k in keys})
            continue

        metrics[f"{name}_silhouette"] = float(silhouette_score(x, y))

        cluster_labels = KMeans(
            n_clusters=n_classes, random_state=kmeans_seed, n_init=10
        ).fit_predict(x)
        metrics[f"{name}_kmeans_ari"] = float(adjusted_rand_score(y, cluster_labels))
        metrics[f"{name}_kmeans_nmi"] = float(
            normalized_mutual_info_score(y, cluster_labels)
        )

        min_class_count = int(class_counts.min())
        folds = min(knn_cv_folds, min_class_count)
        if folds < 2:
            logger.warning(
                "embedding_quality_metrics: label set %r's smallest class "
                "has only %d sample(s) -- cannot cross-validate a kNN "
                "classifier, reporting NaN for %s_knn_accuracy.",
                name,
                min_class_count,
                name,
            )
            metrics[f"{name}_knn_accuracy"] = float("nan")
            continue

        knn = KNeighborsClassifier(n_neighbors=min(knn_k, min_class_count - 1))
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=kmeans_seed)
        metrics[f"{name}_knn_accuracy"] = float(
            np.mean(cross_val_score(knn, x, y, cv=cv))
        )

    return metrics
