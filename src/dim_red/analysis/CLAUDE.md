# CLAUDE.md

Guidance for `src/dim_red/analysis/` (moved out of the repo-root `CLAUDE.md` since it's specific to this directory).

`analysis/` glues the rest of `dim_red` into an end-to-end pipeline. `workflow.run_pca_reduction` fetches structures per crystal system → computes SOAP vectors (species auto-detected across all fetched structures) → standardizes → PCA-reduces, returning `(X_reduced, labels, material_ids)`. `plotting.plot_reduced_space` renders a 2D scatter colored by label; `plotting.plot_spacegroup_histogram` renders a spacegroup-count bar chart colored by crystal family. `examples/run_analysis_demo.py` shows the intended top-level usage.
