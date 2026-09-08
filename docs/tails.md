# Tail training — fase 2

Congela il corpo di un run {bdg-warning}`supcon` / {bdg-success}`cgcnn` / {bdg-secondary}`mace` già completato e allena esattamente un "tail" sopra le rappresentazioni salvate — architettura-agnostico, opera solo su `embeddings.npz` (mai sul corpo stesso, mai su un forward pass ricomputato).

Due percorsi d'uso, stesso motore (`train_tail` sotto):

- **Manuale**: `dimred-train-tail <config> <run_dir>` — schema YAML indipendente (`TailTrainConfig`), pensato per riusare la stessa config su molti run diversi.
- **Automatico**: blocco `tails:` di `RunConfig` (`AutoTailsConfig`) — invocato subito dopo la fase 1, dentro lo stesso `dimred-run`/`dimred-sweep`. Un fallimento qui interrompe il run/sweep (fail-fast).

```{eval-rst}
.. autofunction:: dim_red.pipeline.tail_training.train_tail
```

## `tail_kind = "classification"`

Un'unica head di famiglia (+ spacegroup mascherata su famiglia, opzionale).

```{eval-rst}
.. autoclass:: dim_red.pipeline.config.ClassificationTailConfig

```

## `tail_kind = "visualization"`

Riusa la stessa loss SupCon della fase 1, applicata a una proiezione appresa a bassa dimensionalità pensata per essere *guardata*.

```{eval-rst}
.. autoclass:: dim_red.pipeline.config.VisualizationTailConfig

```

## `tail_kind = "hierarchical"`

Classificatore genuinamente a due stadi: stadio 1 predice la famiglia; stadio 2 è un "esperto" indipendente *per famiglia*, addestrato solo sulle righe di quella famiglia sui soli spacegroup osservati al suo interno. A differenza degli altri tail, permette la classificazione end-to-end di strutture del tutto nuove (`pipeline.inference.predict_hierarchical`).

```{eval-rst}
.. autoclass:: dim_red.pipeline.config.HierarchicalTailConfig


.. autofunction:: dim_red.pipeline.inference.predict_hierarchical
```

## Meccaniche di training condivise

```{eval-rst}
.. autoclass:: dim_red.pipeline.config.TailTrainSettings

```

## Auto-invocazione dalla fase 1 (`RunConfig.tails`)

```{eval-rst}
.. autoclass:: dim_red.pipeline.config.AutoTailsConfig

```

## Config standalone (`dimred-train-tail`)

```{eval-rst}
.. autoclass:: dim_red.pipeline.config.TailTrainConfig

```
