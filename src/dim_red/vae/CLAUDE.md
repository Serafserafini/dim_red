# CLAUDE.md

Guidance for `src/dim_red/vae/` (moved out of the repo-root `CLAUDE.md` since it's specific to this directory).

`vae/` is a separate, JAX/Flax-based neural dimensionality reduction path, independent of the PCA/UMAP/SOAP pipeline (it operates on arbitrary feature matrices via `VAEDatabase`, not specifically SOAP output):

- `model.py`: `VAEModule` (Flax `nn.Module`) composing `EncoderModule`/`DecoderModule` MLPs with configurable hidden-layer sequences; `VAE` is the high-level wrapper holding `params` and exposing `encode`/`decode`/`forward` plus `*_with_params` variants for use inside jitted training steps. Decoder hidden dims default to the mirror (reverse) of the encoder's unless `mirror=False`.
- `codec.py`: `split_encoder_decoder` produces standalone `VAEEncoder`/`VAEDecoder` callables from a trained `VAE` for downstream use without exposing the full model.
- `database.py`: `VAEDatabase` wraps a `(n_samples, n_features)` array and provides a reproducible `train_val_split`.
- `training.py`: `train_vae` runs a full train/val loop using `learned_optimization`'s VeLO meta-learned optimizer (via `prefab.optax_lopt`) as the optimizer backend — note `TrainConfig.learning_rate` is unused by VeLO and kept only for API compatibility. Batches are jitted with `jax.jit`/`jax.vmap`; validation batches are padded to a fixed size with a mask so `vmap` can process a whole epoch in one call. `TrainConfig.device` selects the JAX backend (`"cpu"`/`"gpu"`) and training raises if no matching JAX device exists.

A completed `model: vae` run can have a **visualization tail** trained on top of its frozen `mu` representation via `dimred-train-tail`/`RunConfig.tails.visualization` — see `src/dim_red/pipeline/CLAUDE.md`'s `_TAIL_MODEL_KINDS`. This is the only tail kind offered for `vae` (unlike `supcon`/`cgcnn`/`mace`, a `vae` body already has its own `aux_heads` classification, so a classification/hierarchical tail would be pure duplication for it) — a genuinely useful addition regardless, since `aux_heads` only ever produces a classifier head, never a learned low-dimensional *view* of the latent space the way a visualization tail does. No `vae`-specific code was needed for this: `dim_red.supcon.tail_training.train_visualization_tail` already operates purely on whatever `r_train`/`r_val` arrays `embeddings.npz` happens to contain (here, `mu`, not a body-specific representation), architecture-agnostic from the start.

See the repo-root `CLAUDE.md` for the Optimizer-selection and Early-stopping conventions, which apply here too (duplicated identically across `vae.training`/`autoencoder.training`/`supcon.training`) and for the `dim_red/__init__.py` lazy-import convention this package's own `__init__.py` follows.
