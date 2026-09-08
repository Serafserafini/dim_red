# Training & ottimizzatore

Blocco YAML `train:`, meccaniche di training condivise **identicamente** (codice duplicato, stessi nomi di campo) da `vae.training`, `autoencoder.training`, `supcon.training` (fase 1) e `cgcnn.training`.

```{eval-rst}
.. autoclass:: dim_red.pipeline.config.TrainSettings

```

## Adam vs VeLO

`"adam"` è il default: costruisce un semplice `optax.adam(learning_rate)`. `"velo"` usa l'ottimizzatore meta-appreso pre-addestrato di `learned_optimization` (`prefab.optax_lopt`) — comportamento originale di questa codebase prima che `"adam"` diventasse il default.

Il caricamento di VeLO importa `learned_optimization` e carica il suo checkpoint pre-addestrato: costa alcuni secondi fissi *prima ancora che il training inizi*, e su alcune reti tenta di risolvere credenziali Google Cloud in un modo che può bloccarsi molto più a lungo se quel percorso è lento/irraggiungibile. Questo costo è fisso per chiamata di training a prescindere dalla dimensione del problema — quindi proporzionalmente peggiore per il training dei tail in fase 2 (piccoli MLP a un solo hidden layer, dove un ottimizzatore meta-appreso non porta alcun vantaggio — vedi {doc}`tails`).

```{admonition} Cambio di default deliberato
:class: important

Qualunque config esistente che non imposta esplicitamente `train.optimizer` ora si allena con Adam invece di VeLO. Imposta `train.optimizer: velo` per tornare al comportamento originale.
```

## Early stopping

Implementato identicamente in tutti e quattro i training loop (`vae`, `autoencoder`, `supcon`, `cgcnn`). Monitora sempre `val_loss` — non configurabile su un'altra metrica in questa prima versione.

```{eval-rst}
.. autoclass:: dim_red.pipeline.config.EarlyStoppingConfig

```

```{note}
Per `pipeline.compare`: i grafici/CSV di "metrica finale" leggono `loss_history[metric][-1]` (l'ultima epoca *registrata*), che con `restore_best_weights=True` può essere peggiore di quanto effettivamente ottenuto dai parametri ripristinati qualche epoca prima.
```
