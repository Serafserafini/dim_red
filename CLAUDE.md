# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

This project must be run inside the conda environment `dmred` (located at `/home/seraftu/miniconda3/envs/dmred`). In a persistent terminal session, run `conda activate dmred` once, verify it's active (`$CONDA_DEFAULT_ENV` should equal `dmred`), then run Python/pip/pytest commands directly without prefixing them with `conda run -n dmred`.

There is no `requirements.txt`/`pyproject.toml` dependency list checked in — `pyproject.toml` only holds pytest config. Assume dependencies (numpy, scikit-learn, umap-learn, dscribe, ase, pymatgen, mp-api, jax, flax, optax, learned_optimization, matplotlib) are already installed in the `dmred` environment; don't try to `pip install -e .[dev]` as described in the (stale) README.

## Commands

Run all tests (from repo root, `dmred` active — `pythonpath = ["src"]` in `pyproject.toml` makes the package importable without installation):

```bash
pytest
pytest -v
```

Run a single test file or test:

```bash
pytest tests/test_pca.py
pytest tests/test_pca.py::test_pca_fit_transform -v
```

Some test modules guard optional/heavy dependencies with `pytest.importorskip` (`test_umap.py` skips without `umap`; `test_vae_model.py`/`test_vae_training.py` skip without `jax`) — if those deps aren't in the active env, those tests silently skip rather than fail.

Pre-commit hooks (black, isort with the black profile, trailing-whitespace/EOF/YAML checks) are configured in `.pre-commit-config.yaml`. Run `pre-commit run --all-files` before committing if hooks aren't installed as a git hook.

The `MP_API_KEY` environment variable is required for any code path that calls `dim_red.fetch` / `run_pca_reduction` against the real Materials Project API (see `examples/run_analysis_demo.py`). Tests mock `MPRester`, so they don't need it.

## Architecture

`src/dim_red/` is a materials-science dimensionality-reduction toolkit with two independent verticals joined by an analysis workflow:

- **Data ingestion** (`fetch.py`): pulls crystal structures from the Materials Project API (`mp_api`) filtered by crystal system, converts pymatgen `Structure` objects to ASE `Atoms` via `AseAtomsAdaptor`, and stashes the `material_id` on `atoms.info`.
- **Featurization** (`soap.py`): computes SOAP (Smooth Overlap of Atomic Positions) descriptors from `Atoms` objects using `dscribe`. Has an optional `normalize_distances` preprocessing step that rescales atomic positions so the nearest-neighbor distance is 1.0, making descriptors invariant to absolute bond-length scale.
- **Classical dimensionality reduction** (`pca.py`, `umap.py`, `utils.py`): thin, fit/transform-style wrapper classes around scikit-learn's `PCA` and `umap-learn`'s `UMAP`, both validating `n_components` and raising if used before fitting. `utils.standardize` does z-score normalization and is the shared preprocessing step before either reduction.
- **`analysis/`**: glues the above into an end-to-end pipeline. `workflow.run_pca_reduction` fetches structures per crystal system → computes SOAP vectors (species auto-detected across all fetched structures) → standardizes → PCA-reduces, returning `(X_reduced, labels, material_ids)`. `plotting.plot_reduced_space` renders a 2D scatter colored by label. `examples/run_analysis_demo.py` shows the intended top-level usage.
- **`vae/`**: a separate, JAX/Flax-based neural dimensionality reduction path, independent of the PCA/UMAP/SOAP pipeline (it operates on arbitrary feature matrices via `VAEDatabase`, not specifically SOAP output):
  - `model.py`: `VAEModule` (Flax `nn.Module`) composing `EncoderModule`/`DecoderModule` MLPs with configurable hidden-layer sequences; `VAE` is the high-level wrapper holding `params` and exposing `encode`/`decode`/`forward` plus `*_with_params` variants for use inside jitted training steps. Decoder hidden dims default to the mirror (reverse) of the encoder's unless `mirror=False`.
  - `codec.py`: `split_encoder_decoder` produces standalone `VAEEncoder`/`VAEDecoder` callables from a trained `VAE` for downstream use without exposing the full model.
  - `database.py`: `VAEDatabase` wraps a `(n_samples, n_features)` array and provides a reproducible `train_val_split`.
  - `training.py`: `train_vae` runs a full train/val loop using `learned_optimization`'s VeLO meta-learned optimizer (via `prefab.optax_lopt`) as the optimizer backend — note `TrainConfig.learning_rate` is unused by VeLO and kept only for API compatibility. Batches are jitted with `jax.jit`/`jax.vmap`; validation batches are padded to a fixed size with a mask so `vmap` can process a whole epoch in one call. `TrainConfig.device` selects the JAX backend (`"cpu"`/`"gpu"`) and training raises if no matching JAX device exists.
- **`autoencoder/`**: a deterministic (non-variational) counterpart to `vae/`, mirroring its exact module layout and public API shape (`model.py`'s `Autoencoder`/`Encoder`/`Decoder`, `codec.py`'s `split_encoder_decoder`, `training.py`'s `TrainConfig`/`train_autoencoder`) so the two are interchangeable. The only real difference: `Autoencoder.encode` returns a single deterministic `z` (not a sampled `(mu, logvar)` posterior), so there's no KL term, no `beta`, and no PRNG key needed at encode/decode/forward time. Reuses `vae.database.VAEDatabase` directly (dataset handling has nothing model-specific about it) rather than duplicating it. `pipeline.single_run.run_single` picks between `vae`/`autoencoder` via `RunConfig.model_kind` (YAML: top-level `model: vae` or `model: autoencoder`, default `vae`); both read architecture from the same `vae:` config block and aux-heads from the same `aux_heads:` block, so a sweep can vary `model` like any other grid axis (see `configs/single_run_autoencoder.example.yaml`).

`dim_red/__init__.py` lazily imports `soap`, `fetch`, `analysis`, `vae`, and `autoencoder` via module-level `__getattr__` (PEP 562) so that importing the base package doesn't require optional heavy dependencies (dscribe, mp_api, jax, flax, etc.) unless those submodules are actually used. `vae/__init__.py` and `autoencoder/__init__.py` do the same lazy pattern internally for their own symbols. When adding new public symbols to any of these packages, follow this lazy-import pattern rather than importing eagerly at module top.
