# Tail cgcnn (`dimred-train-tail`)

Congela il corpo di un run {bdg-success}`cgcnn` già completato e allena esattamente un "tail" sopra le rappresentazioni salvate — opera solo su `embeddings.npz` (mai sul corpo stesso).

```{note}
Per `supcon`/`supcon_mace` le heads (classificatore + visualizzatore, e gli esperti per spacegroup) fanno parte di `FullStack` e si allenano con `dimred-train-heads`: vedi {doc}`fullstack`. Il blocco `tails:` di `RunConfig`, `tail_kind: hierarchical_supcon` e i campi `sg_*` non esistono più (`tails:` in una config è un errore esplicito).
```

`dimred-train-tail <config> <run_dir>` — schema YAML indipendente (`TailTrainConfig`), pensato per riusare la stessa config su molti run diversi. Un run non cgcnn viene rifiutato.

```{eval-rst}
.. autofunction:: dim_red.pipeline.tail_training.train_tail
```

## `tail_kind = "classification"`

Un classificatore di famiglia (MLP con cross-entropy).

```{eval-rst}
.. autoclass:: dim_red.pipeline.config.ClassificationTailConfig

```

## `tail_kind = "visualization"`

Riusa la stessa loss SupCon della fase 1, applicata a una proiezione appresa a bassa dimensionalità pensata per essere *guardata*.

```{eval-rst}
.. autoclass:: dim_red.pipeline.config.VisualizationTailConfig

```

## Meccaniche di training condivise

```{eval-rst}
.. autoclass:: dim_red.pipeline.config.TailTrainSettings

```

## Config standalone

```{eval-rst}
.. autoclass:: dim_red.pipeline.config.TailTrainConfig

```
