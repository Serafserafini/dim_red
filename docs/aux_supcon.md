# Head ausiliarie & batching

## `aux_heads:` — classificatori sulla z

Usato solo da {bdg-success}`cgcnn` (per cui è obbligatorio, `mode != "none"`): in CGCNN la classificazione è l'unico obiettivo di training.

```{eval-rst}
.. autoclass:: dim_red.pipeline.config.AuxHeadsConfig

```

## Loss contrastiva di supcon / supcon_mace

La loss Supervised Contrastive (Khosla et al. 2020) è calcolata sull'output di una `ProjectionTail`, non sulla rappresentazione `r` del corpo. I suoi parametri (`tau`, `distance`, `lambda_norm`) stanno nei blocchi `contrastive:` e `viz:` di ogni stack di FullStack — vedi {doc}`fullstack`.

## `batching:` — campionamento dei batch

Solo i batch di *training* sono interessati — quelli di validazione restano sempre casuali. Nelle config FullStack il blocco `batching:` di uno stack ha la stessa forma; qui sotto la dataclass usata dal tail di visualizzazione cgcnn.

```{eval-rst}
.. autoclass:: dim_red.pipeline.config.BatchingConfig


.. autoclass:: dim_red.pipeline.config.BalancedBatchingParams

```
