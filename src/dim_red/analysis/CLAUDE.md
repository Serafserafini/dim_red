# CLAUDE.md

Guidance for `src/dim_red/analysis/`.

- `plotting.py`: `plot_reduced_space`/`plot_reduced_space_3d` (scatter colored by label, neutral "Dimension N" axis defaults), `plot_applied_structures` (overlay new points), `plot_spacegroup_histogram` (spacegroup counts colored by crystal family), `plot_confusion_matrix`, `plot_classification_report`, `plot_reliability_diagram`.
- `metrics.py`: `embedding_quality_metrics` — sklearn-only silhouette / k-means ARI+NMI / kNN accuracy per label set; the scoring function behind `pipeline.benchmark`. Quantifies quality, never picks a "best" run.
