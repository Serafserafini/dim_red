# RunConfig di primo livello & sweep

## RunConfig

```{eval-rst}
.. autoclass:: dim_red.pipeline.config.RunConfig
```

### Regole di validazione (`__post_init__`)

- `model_kind` deve essere uno tra `supcon`, `supcon_mace`, `cgcnn` (`vae`, `autoencoder` e `mace` sono stati rimossi).
- `data_source` deve essere `"fetch"` o `"pyxtal"`.
- `data_source="fetch"` richiede il blocco `fetch`; `data_source="pyxtal"` richiede il blocco `pyxtal`.
- `model_kind="cgcnn"` richiede `aux_heads.mode != "none"` — è la sua unica sorgente di training.
- `model_kind="supcon_mace"` richiede `mace.checkpoint_path` non vuoto — le sue feature di ingresso vengono dal forward pass di un modello MACE congelato.

Un campo YAML sconosciuto per una sotto-config viene ignorato con un `logger.warning` (non un errore fatale) — utile per intercettare refusi senza bloccare l'intero run.

### Matrice di compatibilità model_kind × blocco config

```{list-table}
:header-rows: 1
:widths: 28 24 24 24

* - Blocco
  - supcon
  - supcon_mace
  - cgcnn
* - `soap`
  - ✓
  - –
  - –
* - `graph`
  - –
  - –
  - ✓
* - `mace`
  - –
  - ✓ richiesto
  - –
* - `encoder`
  - ✓ intero
  - ✓ intero
  - solo `latent_dim`
* - `aux_heads`
  - ignorato
  - ignorato
  - ✓ richiesto
* - `supcon`
  - ✓
  - ✓
  - ignorato
* - `batching`
  - ✓
  - ✓
  - ignorato
* - `tails`
  - ✓
  - ✓
  - ✓
* - Fasi di training
  - 1 + tail opz.
  - 1 + tail opz.
  - 1
```

## Sweep — grid search

`SweepConfig` combina un `base` (dizionario con la stessa forma nidificata di `RunConfig`) con un `grid`: assi a percorso puntato (es. `"encoder.latent_dim"`, `"train.learning_rate"`, `"aux_heads.lambda_family"`, `"seed"`) → lista di valori. `expand_sweep` genera un `RunConfig` per ogni combinazione del prodotto cartesiano di tutti gli assi. **Qualunque** campo di `RunConfig` è sweepabile così — non un insieme fisso di assi nominati.

```{eval-rst}
.. autoclass:: dim_red.pipeline.config.SweepConfig
   :members: output_dir, api_key

.. autofunction:: dim_red.pipeline.config.expand_sweep

.. autofunction:: dim_red.pipeline.sweep.run_sweep
```
