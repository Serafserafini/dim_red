# Tail training — fase 2

Congela il corpo di un run {bdg-warning}`supcon` / {bdg-secondary}`supcon_mace` / {bdg-success}`cgcnn` già completato e allena esattamente un "tail" sopra le rappresentazioni salvate — architettura-agnostico, opera solo su `embeddings.npz` (mai sul corpo stesso, mai su un forward pass ricomputato).

Due percorsi d'uso, stesso motore (`train_tail` sotto):

- **Manuale**: `dimred-train-tail <config> <run_dir>` — schema YAML indipendente (`TailTrainConfig`), pensato per riusare la stessa config su molti run diversi.
- **Automatico**: blocco `tails:` di `RunConfig` (`AutoTailsConfig`) — invocato subito dopo la fase 1, dentro lo stesso `dimred-run`/`dimred-sweep`. Un fallimento qui interrompe il run/sweep (fail-fast).

```{eval-rst}
.. autofunction:: dim_red.pipeline.tail_training.train_tail
```

## `tail_kind = "classification"`

Un classificatore di famiglia (MLP con cross-entropy). La classificazione dello spacegroup non si fa qui: la fanno gli esperti per famiglia di `hierarchical_supcon`.

```{eval-rst}
.. autoclass:: dim_red.pipeline.config.ClassificationTailConfig

```

## `tail_kind = "visualization"`

Riusa la stessa loss SupCon della fase 1, applicata a una proiezione appresa a bassa dimensionalità pensata per essere *guardata*.

```{eval-rst}
.. autoclass:: dim_red.pipeline.config.VisualizationTailConfig

```

## `tail_kind = "hierarchical_supcon"`

Classificatore a due stadi in cui ogni stadio ha lo stesso aspetto del modello a livello di famiglia (encoder + proiezione, poi classificatore e visualizzatore). Stadio 1: un classificatore di famiglia sull'embedding del corpo. Stadio 2: per ogni famiglia, un *esperto* — un nuovo encoder SupCon con la sua proiezione, allenato solo sulle righe di quella famiglia (sulle feature native: SOAP per `supcon`, embedding MACE per `supcon_mace`) con loss contrastiva sullo spacegroup locale, più un classificatore e un visualizzatore 2D sull'embedding congelato dell'esperto. Le famiglie con troppi pochi campioni o meno di 2 spacegroup non hanno un esperto e prevedono lo spacegroup più frequente.

Non è disponibile tramite `RunConfig.tails`: si lancia con `dimred-train-tail`.

```{eval-rst}
.. autoclass:: dim_red.pipeline.config.HierarchicalSupconTailConfig

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
