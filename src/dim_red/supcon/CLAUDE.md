# CLAUDE.md

Guidance for `src/dim_red/supcon/`.

Encoder-only Supervised Contrastive (Khosla et al. 2020) stack: a **body** (`SupConEncoder`, producing a representation `r`) plus one attachable **tail** at a time. Also used unchanged for `model_kind: supcon_mace` (the only difference is the input: frozen MACE embeddings instead of SOAP) — never touch the `in ("supcon", "supcon_mace")` checks in `pipeline/`.

- `model.py`: `EncoderModule` (Dense+ReLU MLP) and `SupConEncoder` (`params`, `encode`/`encode_with_params`).
- `tails.py`: `MLP` (width `hidden_dim` int or list) shared by `_MLPTail` → `ProjectionTail` (phase 1, output width `projection_dim`) and `VisualizationTail` (phase 2, output 2 or 3); plus `ClassificationTail(input_dim, hidden_dim, n_classes, seed)` with `classify`/`classify_with_params`. `ClassificationTail.load_params_bytes` accepts legacy checkpoints saved as `{"family_head": {...}}`. Every tail has its own params pytree, never nested in the body's — freezing the body in phase 2 is just leaving it out of the optimized dict. Flax names layers `Dense_N` by creation order, so checkpoints stay compatible across the MLP unification.
- `training.py` (**phase 1**): `TrainConfig` (no lambdas — those live in `pipeline.config.SupConConfig`), `training_first_phase(...)` jointly trains body + `ProjectionTail` on `z = Proj(r)`, `supcon_loss` (similarity: negative squared Euclidean or cosine, `tau`), `norm_penalty` (optional, `lambda_norm`). Anchors with no positive in the batch are excluded from the mean; self-comparisons are masked with a large finite negative (not `-inf`, `0 * -inf` is NaN). Validation uses a plain unpadded batch loop since SupCon is pairwise within a batch. Optimizer is always Adam (`learning_rate`). Early stopping on `val_loss` (see root `CLAUDE.md`).
- `tail_training.py` (**phase 2**): `TailTrainConfig` (an alias of `training.TrainConfig`), `train_classification_tail` (cross-entropy, single head; history keys `train_ce`/`val_ce`) and `train_visualization_tail` (SupCon on the tail's own output, frozen `r`).
- `sampling.py`: `"balanced"` batching (P families × K examples, stratified across spacegroups) for training batches only.

Design principle: family-level models and per-family experts are the same encoder + projection + visualization + classification stack; the expert path is `pipeline.tail_training`'s `hierarchical_supcon`. Further unification is tracked in `REFACTOR_TODO.md`. Run entry points: `src/dim_red/pipeline/CLAUDE.md`.
