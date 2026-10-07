# CLAUDE.md

Guidance for `src/dim_red/pipeline/`. User-facing description of commands, config and layout: `docs/fullstack.md`.

**Two config paths.** `model_kind: supcon|supcon_mace` = FullStack (`full_stack_config.py`: strict YAML, `family` + `experts.defaults`/per-expert overrides, resolved into one `StackSpec` per stack; pyxtal only). `RunConfig` (`config.py`) / `run_single` are **cgcnn-only** (`model: cgcnn`; `tails:` and non-cgcnn `model:` raise explicit errors; removed keys such as `optimizer`/`beta`/`decoder_hidden_dim` and the old `supcon`/`batching`/`mace` blocks are ignored; the legacy `vae:` block name is merged under `encoder:`). The optimizer is always Adam. Dataset cache keys pin the removed fields to their old defaults so existing caches stay valid.

Entry points (`cli.py`): `dimred-run` (dispatches on config keys: `model_kind`/`family`/`experts` -> FullStack, else cgcnn), `dimred-rerun` (FullStack: `stacks` key in the saved config; creates a new run dir), `dimred-sweep` (FullStack only, writes `sweep.yaml`), `dimred-train-heads <config> <run_dir> --heads-name NAME [--stacks a,b]` (only classifier/viz/seed of each listed stack are read), `dimred-compare`, `dimred-benchmark`, `dimred-apply`, `dimred-train-tail` (cgcnn only; `tail_training._TAIL_MODEL_KINDS`).

**`full_stack.py`**: `FullStack.create/open`, `fit_body(stacks=None)`, `fit_heads(name, stacks=None, config=None)`, `predict`/`predict_stack`, `apply_to_structures`. Each stack: `build_stack_dataset` (`stack_data.py`, reuses `dataset_cache`) -> grouped split -> `SingleStack`. Per-stack seed is `run_seed + index in STACK_ORDER` unless set. `fit_body` onto an existing `stacks/<name>/` raises `FileExistsError`; too few training rows or fewer than 2 spacegroups in an expert is an error (no fallback). A failing stack stops the run and names the stack. `predict`: family stack picks the system, then that system's expert; no trained expert -> `None` fields; `device="cpu"` default always works for GPU-trained runs.

**`run_layout.py`** (jax-free, used by compare/benchmark): `stacks/<name>/{config.yaml, dataset.extxyz, classes.yaml, embeddings.npz, body/, heads/<name>/}`; `.<x>.tmp` dirs are interrupted atomic writes and never listed; `resolve_heads_name` never picks silently among several heads sets (`--heads-name`). **`featurize.py`**: features of new structures (SOAP `mu2` or MACE) + `standardize` with the stack's saved mean/std. Old (pre-FullStack) run layouts are not readable.

**Train/val split is grouped by `material_id`** (`_common._split_indices_grouped`), because augmented copies share `material_id` with their original; a per-row split leaks near-duplicates into validation (held-out pyxtal sets showed family accuracy ~0.95 val vs ~0.5 held out; see `examples/evaluate_holdout_pyxtal.py`). Without duplicates it degenerates to the old per-row split. The family and expert datasets share no structures, so chain evaluation needs a held-out set. Phase-2 reads the split back from `embeddings.npz["split"]`.

**`compare`/`benchmark`** are FullStack-only: one plot suite / one CSV row per (run, stack). Accuracies are val-only under the bare `family`/`spacegroup` keys when the npz has a `split` array, with `*_all`/`*_train`/`*_val` alongside; old benchmark CSVs used pooled numbers, so don't compare bare columns across them. They never rank or pick a "best" run. `*_family_supcon` history keys also exist in experts (they contrast spacegroups there; plots relabel them).

**`inference.py`** (`load_trained_run`, `apply_model_to_structures`) and `tail_training.py` serve cgcnn runs only (`src/dim_red/cgcnn/CLAUDE.md`).

See the repo-root `CLAUDE.md` for the early-stopping convention, `src/dim_red/supcon/CLAUDE.md` for `SingleStack`, `src/dim_red/mace/CLAUDE.md` for MACE.
