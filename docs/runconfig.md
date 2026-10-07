# RunConfig (cgcnn) & sweep

`RunConfig` / `run_single` allenano solo `cgcnn` (`model: cgcnn`). Per `supcon`/`supcon_mace` lo schema di config, i comandi e le sweep sono descritti in {doc}`fullstack`.

## RunConfig

```{eval-rst}
.. autoclass:: dim_red.pipeline.config.RunConfig
```

### Regole di validazione (`__post_init__`)

- `model_kind` deve essere `cgcnn`; `supcon`/`supcon_mace` usano lo schema FullStack e `model:` diverso da `cgcnn` è un errore esplicito.
- `data_source` deve essere `"fetch"` o `"pyxtal"`.
- `data_source="fetch"` richiede il blocco `fetch`; `data_source="pyxtal"` richiede il blocco `pyxtal`.
- `model_kind="cgcnn"` richiede `aux_heads.mode != "none"` — è la sua unica sorgente di training.

I blocchi `supcon`, `batching` e `mace` di config cgcnn salvate in precedenza vengono ignorati; il blocco top-level `tails:` è un errore esplicito. Un campo YAML sconosciuto per una sotto-config viene ignorato con un `logger.warning` (non un errore fatale).

```{note}
Il parser di FullStack è invece stretto: una config vecchia (per esempio con `model: supcon`, `tails:` o i campi `sg_*`) viene rifiutata, e `tests/test_example_configs.py` carica ogni config in `configs/`.
```

## Sweep

`dimred-sweep` esegue solo sweep FullStack: `SweepConfig` combina un `base` (dizionario con la forma nidificata di una config FullStack) con un `grid` di percorsi puntati (`family.encoder.latent_dim`, `experts.defaults.train.epochs`, `seed`, ...) → lista di valori, e `run_sweep` esegue una run (body + heads `default`) per ogni combinazione del prodotto cartesiano, scrivendo `sweep.yaml` nella cartella della sweep. Le config cgcnn si lanciano con `dimred-run`.

```{eval-rst}
.. autoclass:: dim_red.pipeline.config.SweepConfig
   :members: output_dir

.. autofunction:: dim_red.pipeline.sweep.run_sweep
```
