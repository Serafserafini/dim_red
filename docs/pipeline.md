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
supcon · supcon_mace (FullStack) · cgcnn
`model_kind`
+++
{doc}`model_kinds`
:::

:::{grid-item-card} 5 · Heads / tail (fase 2)
:class-card: sd-border-dashed
FullStack: classifier + viz per stack · cgcnn: `dimred-train-tail`
`dimred-train-heads`
+++
{doc}`fullstack` · {doc}`tails`
:::

:::{grid-item-card} 6 · Analisi
compare · benchmark · apply
`pipeline.*`
+++
{doc}`tools`
:::

::::

`supcon`/`supcon_mace` si allenano solo con `FullStack` (schema di config, comandi e layout su disco in {doc}`fullstack`); `RunConfig`/`run_single` sono solo `cgcnn`. `dimred-run` sceglie in base alle chiavi della config. `dimred-sweep` (solo FullStack) esegue una run per ogni combinazione del prodotto cartesiano definito in `grid` (vedi {doc}`runconfig`); `pipeline.benchmark` confronta run in un'unica tabella, una riga per stack (vedi {doc}`tools`).

## Le config posizionali di un run cgcnn

Un {py:class}`~dim_red.pipeline.config.RunConfig` (riferimento completo in {doc}`runconfig`) richiede sempre:

- `soap` — blocco obbligatorio dello schema, anche se `cgcnn` usa `graph`.
- `encoder` — `cgcnn` ne legge solo `latent_dim`.
- `train` — meccaniche di training condivise.

La {doc}`runconfig` raccoglie anche le regole di validazione (`__post_init__`).
