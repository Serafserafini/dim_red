# CLAUDE.md

Guidance for `src/dim_red/cgcnn/`.

JAX/Flax reimplementation of Crystal Graph Convolutional Neural Networks (Xie & Grossman 2018). No SOAP: it builds its own bond graph from `Atoms`. Unlike `supcon`, classification is trained jointly inside the body (cross-entropy on `aux_heads`), so `aux_heads.mode != "none"` is required.

Deliberate deviations from the paper:
- Atom features are a per-structure LOCAL species slot (distinct atomic numbers sorted ascending → slots `1..k`, `0` = padding), so the model never sees real chemical identity.
- `nn.LayerNorm` instead of `BatchNorm1d` (no mutable batch stats; params stay one pure pytree).

Files:
- `graph.py` (NumPy + ASE, no jax): `ase.neighborlist.neighbor_list` for periodic-correct neighbors (cutoffs may exceed half the cell width), Gaussian-expanded distances, padded arrays (`atoms_to_graph`, `atoms_list_to_graph_arrays`).
- `database.py`: `GraphDatabase` — five parallel arrays, `train_val_split` (same API as `dim_red.dataset.FeatureDatabase`; dedup tracked in `REFACTOR_TODO.md`).
- `model.py`: `ConvLayer` (batched gather, gated conv), `CGCNNBodyModule`, `CGCNNEncoderModule`, `CGCNNEncoder` (`encode`, `classify_family`, `classify_spacegroup`, `*_with_params`), `ClassifierHead`, `apply_family_mask`.
- `training.py`: `TrainConfig`, `train_cgcnn` (Adam, early stopping, padded-batch `vmap` evaluation).

Pipeline integration: `GraphConfig` (`graph:` block; `latent_dim` is read from `encoder.latent_dim`), `dataset_cache.build_graph_dataset_for_run`, an `is_cgcnn` branch in `single_run`/`inference`. `embeddings.npz` has no `features`/`feature_mean`/`feature_std` for cgcnn. Phase-2 tails (`classification`, `visualization`) can be trained on top of it. See `configs/single_run_cgcnn.example.yaml`.
