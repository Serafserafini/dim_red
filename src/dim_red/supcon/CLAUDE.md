# CLAUDE.md

Guidance for `src/dim_red/supcon/`.

Encoder-only Supervised Contrastive (Khosla et al. 2020) stack: a **body** (`SupConEncoder`, representation `r`) plus a projection tail trained with it (phase 1), then a classification tail and a visualization tail trained on the frozen `r` (phase 2). Also used for `model_kind: supcon_mace` (input = frozen MACE embeddings instead of SOAP). Only `FullStack` (`pipeline/full_stack.py`) trains these; the family level and every per-system expert are the same `SingleStack`.

- `stack.py`: `StackConfig` (`body_train`/`classifier_train`/`viz_train`, widths, batching, seed), `SingleStack` (arrays and models only, no I/O beyond `save_body`/`load_body`/`save_heads`/`load_heads`). `fit_body(X_train, X_val, y_train, y_val, sub_train=None)` calls `training_first_phase`; `fit_heads(...)` freezes the encoder and trains classifier + viz; `encode`, `predict_proba`, `visualize`. The label is one integer array (crystal system or spacegroup, decided by the role). `load_*` take `device=` so GPU-trained params load on CPU.
- `model.py`: `EncoderModule` (Dense+ReLU MLP), `SupConEncoder` (`params`, `encode`/`encode_with_params`).
- `tails.py`: `MLP` shared by `ProjectionTail` (output `projection_dim`) and `VisualizationTail` (output 2 or 3), plus `ClassificationTail`. Each tail has its own params pytree; freezing the body is leaving it out of the optimized dict. Flax names layers `Dense_N` by creation order, so checkpoints stay compatible across MLP changes.
- `training.py` (phase 1): `TrainConfig`, `training_first_phase(...)`, `supcon_loss` (negative squared Euclidean or cosine similarity, `tau`), `norm_penalty`. Anchors with no positive in the batch are excluded; self-comparisons are masked with a large finite negative (not `-inf`, `0 * -inf` is NaN). Validation is a plain unpadded batch loop (SupCon is pairwise within a batch). Adam only; early stopping on `val_loss` (root `CLAUDE.md`).
- `tail_training.py` (phase 2): `train_classification_tail` (cross-entropy, history keys `train_ce`/`val_ce`), `train_visualization_tail` (SupCon on the tail's own output). `training_first_phase` and `train_visualization_tail` are near-copies; unifying them is in `REFACTOR_TODO.md`.
- `sampling.py`: `"balanced"` batching (P families x K examples, stratified across spacegroups) for training batches only.

Never touch the `in ("supcon", "supcon_mace")` model-kind checks in `pipeline/`. Pipeline side: `src/dim_red/pipeline/CLAUDE.md`.
