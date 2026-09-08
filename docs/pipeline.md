# Mappa della pipeline

Ogni run attraversa queste tappe, in ordine. Le tappe con bordo tratteggiato sono opzionali.

::::{grid} 1 2 3 3
:gutter: 2

:::{grid-item-card} 1 · Sorgente dati
`fetch` oppure `pyxtal`
`RunConfig.data_source`
+++
{doc}`data_sources`
:::

:::{grid-item-card} 2 · Augmentation
:class-card: sd-border-dashed
jitter + vacanze, opzionale
`RunConfig.augmentation`
+++
{doc}`data_sources`
:::

:::{grid-item-card} 3 · Featurizzazione
SOAP · grafo · MACE
`soap` · `graph` · `mace`
+++
{doc}`featurization`
:::

:::{grid-item-card} 4 · Corpo del modello
vae · ae · supcon · cgcnn · mace
`RunConfig.model_kind`
+++
{doc}`model_kinds`
:::

:::{grid-item-card} 5 · Tail (fase 2)
:class-card: sd-border-dashed
classification · visualization · hierarchical
`RunConfig.tails`
+++
{doc}`tails`
:::

:::{grid-item-card} 6 · Analisi
compare · benchmark · apply
`pipeline.*`
+++
{doc}`tools`
:::

::::

Un `SweepConfig` esegue questa stessa pipeline una volta per ogni combinazione del prodotto cartesiano definito in `grid` (vedi {doc}`runconfig`); `pipeline.benchmark` confronta run di `model_kind` diversi in un'unica tabella (vedi {doc}`tools`).

## Le due config posizionali, sempre presenti

Qualunque `model_kind`, un {py:class}`~dim_red.pipeline.config.RunConfig` (riferimento completo in {doc}`runconfig`) richiede sempre:

- `soap` — anche quando non è la featurizzazione effettivamente usata (`cgcnn`/`mace` la ignorano a favore di `graph`/`mace`).
- `vae` — architettura encoder/decoder condivisa (`autoencoder`, `supcon` ne leggono un sottoinsieme; `cgcnn` solo `latent_dim`; `mace` la ignora del tutto).
- `train` — meccaniche di training condivise (per `mace`, che non allena nulla, solo `device` ha senso).

Il resto della configurazione — quale sotto-config si applica a quale `model_kind` — è coperto pagina per pagina; la {doc}`runconfig` raccoglie anche la matrice di compatibilità completa e le regole di validazione (`__post_init__`).
