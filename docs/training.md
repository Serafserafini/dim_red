# Training

Blocco YAML `train:`, meccaniche di training con gli stessi nomi di campo in `cgcnn` (`TrainSettings` qui sotto) e in ogni stack di FullStack (blocco `train:` dello stack, che popola `StackConfig`/`TrainConfig` di `supcon.training`; vedi {doc}`fullstack`). L'ottimizzatore è sempre Adam (`optax.adam(learning_rate)`).

```{eval-rst}
.. autoclass:: dim_red.pipeline.config.TrainSettings

```

```{note}
Le chiavi `train.optimizer` e `train.beta` (ottimizzatore VeLO e peso KL del VAE, entrambi rimossi) possono ancora comparire in config e `config.yaml` salvati: vengono ignorate senza errore.
```

## Early stopping

Implementato identicamente nei training loop di `supcon.training` (body), `supcon.tail_training` (heads) e `cgcnn.training`. Monitora sempre `val_loss` — non configurabile su un'altra metrica in questa prima versione.

```{eval-rst}
.. autoclass:: dim_red.pipeline.config.EarlyStoppingConfig

```

```{note}
Per `pipeline.compare`: i grafici/CSV di "metrica finale" leggono `loss_history[metric][-1]` (l'ultima epoca *registrata*), che con `restore_best_weights=True` può essere peggiore di quanto effettivamente ottenuto dai parametri ripristinati qualche epoca prima.
```
