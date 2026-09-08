# Head ausiliarie & loss contrastiva

Due schemi alternativi per usare le etichette (famiglia cristallina, spacegroup) durante il training del corpo — mutuamente esclusivi in base a `model_kind`.

## `aux_heads:` — classificatori sulla z

Usato da {bdg-primary}`vae` {bdg-info}`autoencoder` {bdg-success}`cgcnn` (per cui è obbligatorio, `mode != "none"`).

```{eval-rst}
.. autoclass:: dim_red.pipeline.config.AuxHeadsConfig

```

## `supcon:` — Supervised Contrastive

Usato solo da {bdg-warning}`supcon`. La loss è calcolata sull'output di una `ProjectionTail` (Khosla et al. 2020), non sulla rappresentazione `r` del corpo stesso — nessuna head di classificazione a questo stadio (quelle arrivano in fase 2, vedi {doc}`tails`).

```{eval-rst}
.. autoclass:: dim_red.pipeline.config.SupConConfig

```

## `batching:` — campionamento dei batch

Usato solo da {bdg-warning}`supcon`. Solo i batch di *training* sono interessati — quelli di validazione restano sempre casuali.

```{eval-rst}
.. autoclass:: dim_red.pipeline.config.BatchingConfig


.. autoclass:: dim_red.pipeline.config.BalancedBatchingParams

```
