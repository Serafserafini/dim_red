# CLAUDE.md

Guidance for `src/dim_red/autoencoder/` (moved out of the repo-root `CLAUDE.md` since it's specific to this directory).

`autoencoder/` is a deterministic (non-variational) counterpart to `vae/`, mirroring its exact module layout and public API shape (`model.py`'s `Autoencoder`/`Encoder`/`Decoder`, `codec.py`'s `split_encoder_decoder`, `training.py`'s `TrainConfig`/`train_autoencoder`) so the two are interchangeable. The only real difference: `Autoencoder.encode` returns a single deterministic `z` (not a sampled `(mu, logvar)` posterior), so there's no KL term, no `beta`, and no PRNG key needed at encode/decode/forward time. Reuses `vae.database.VAEDatabase` directly (dataset handling has nothing model-specific about it) rather than duplicating it. `pipeline.single_run.run_single` picks between `vae`/`autoencoder` via `RunConfig.model_kind` (YAML: top-level `model: vae` or `model: autoencoder`, default `vae`); both read architecture from the same `vae:` config block and aux-heads from the same `aux_heads:` block, so a sweep can vary `model` like any other grid axis (see `configs/single_run_autoencoder.example.yaml`).

Like `vae`, a completed `model: autoencoder` run can have a **visualization tail** trained on top of its frozen `z` via `dimred-train-tail`/`RunConfig.tails.visualization` (not classification/hierarchical, redundant with its own `aux_heads`) -- see `src/dim_red/pipeline/CLAUDE.md`'s `_TAIL_MODEL_KINDS` and `src/dim_red/vae/CLAUDE.md`'s identical note.

See the repo-root `CLAUDE.md` for the Optimizer-selection and Early-stopping conventions, which apply here too (duplicated identically across `vae.training`/`autoencoder.training`/`supcon.training`).
