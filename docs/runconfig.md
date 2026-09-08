# RunConfig di primo livello & sweep

## RunConfig

```{eval-rst}
.. autoclass:: dim_red.pipeline.config.RunConfig
```

### Regole di validazione (`__post_init__`)

- `model_kind` deve essere uno dei 5 valori validi.
- `data_source` deve essere `"fetch"` o `"pyxtal"`.
- `data_source="fetch"` richiede il blocco `fetch`; `data_source="pyxtal"` richiede il blocco `pyxtal`.
- `model_kind="cgcnn"` richiede `aux_heads.mode != "none"` — è la sua unica sorgente di training.
- `model_kind="mace"` richiede `mace.checkpoint_path` non vuoto — un corpo congelato non ha altro da cui costruire i propri pesi.

Un campo YAML sconosciuto per una sotto-config viene ignorato con un `logger.warning` (non un errore fatale) — utile per intercettare refusi senza bloccare l'intero run.

### Matrice di compatibilità model_kind × blocco config

```{list-table}
:header-rows: 1
:widths: 22 13 15 13 13 13

* - Blocco
  - vae
  - autoencoder
  - supcon
  - cgcnn
  - mace
* - `soap`
  - ✓
  - ✓
  - ✓
  - –
  - –
* - `graph`
  - –
  - –
  - –
  - ✓
  - –
* - `mace`
  - –
  - –
  - –
  - –
  - ✓
* - `vae` (encoder/decoder)
  - ✓ intero
  - ✓ intero
  - solo encoder
  - solo `latent_dim`
  - –
* - `train.beta`
  - ✓
  - ignorato
  - n/a
  - n/a
  - n/a
* - `aux_heads`
  - ✓
  - ✓
  - ignorato
  - ✓ richiesto
  - n/a
* - `supcon`
  - ignorato
  - ignorato
  - ✓
  - ignorato
  - n/a
* - `batching`
  - ignorato
  - ignorato
  - ✓
  - ignorato
  - n/a
* - `tails`
  - ignorato
  - ignorato
  - ✓
  - ✓
  - ✓ unico modo
* - Fasi di training
  - 1
  - 1
  - 1 + tail opz.
  - 1
  - nessuna
```

## Sweep — grid search

`SweepConfig` combina un `base` (dizionario con la stessa forma nidificata di `RunConfig`) con un `grid`: assi a percorso puntato (es. `"vae.latent_dim"`, `"train.learning_rate"`, `"aux_heads.lambda_family"`, `"seed"`) → lista di valori. `expand_sweep` genera un `RunConfig` per ogni combinazione del prodotto cartesiano di tutti gli assi. **Qualunque** campo di `RunConfig` è sweepabile così — non un insieme fisso di assi nominati.

```{eval-rst}
.. autoclass:: dim_red.pipeline.config.SweepConfig
   :members: output_dir, api_key

.. autofunction:: dim_red.pipeline.config.expand_sweep

.. autofunction:: dim_red.pipeline.sweep.run_sweep
```
