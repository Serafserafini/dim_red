# dim_red

**dim_red** trasforma database di strutture cristalline in rappresentazioni a bassa dimensionalità, confrontando più famiglie di modelli — variational, deterministico, contrastivo, a grafo, foundation model congelato — su come separano famiglia cristallina e spacegroup.

Questa guida è la mappa di riferimento della codebase: architettura, cosa fa ogni `model_kind`, e — soprattutto — **cosa fa ogni singolo parametro di configurazione**. Le pagine sulla configurazione non ripetono a mano quello che il codice già dice: incorporano i docstring delle dataclass di `pipeline/config.py` tramite `autodoc`, quindi restano sincronizzate quando il codice cambia. La prosa di contorno (perché un default è quel default, come due parametri interagiscono, quando usare cosa) è scritta a mano.

```{admonition} Cosa non c'è qui
:class: scope-note

Nessun numero di run reale, nessun grafico di risultati — `runs/` ed `experiments/` sono cartelle di output vuote in questo momento. Questa guida documenta il **comportamento del codice**, non l'esito di esperimenti già fatti. Quando iniziamo a confrontare run reali, quella è una sezione a parte da aggiungere qui.
```

## Percorsi di lettura

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Sto configurando un run
:link: pipeline
:link-type: doc
Parti dalla mappa della pipeline, poi segui sorgenti dati → featurizzazione → model kind → training.
:::

:::{grid-item-card} Sto scegliendo un model_kind
:link: model_kinds
:link-type: doc
Confronto diretto tra vae, autoencoder, supcon, cgcnn, mace: cosa allenano e quali blocchi YAML usano.
:::

:::{grid-item-card} Sto cercando un parametro preciso
:link: runconfig
:link-type: doc
`RunConfig` di primo livello, con link a ogni sotto-config. Oppure usa la ricerca full-text della sidebar.
:::

:::{grid-item-card} Voglio confrontare run già fatti
:link: tools
:link-type: doc
`pipeline.compare` / `pipeline.benchmark` e cosa significano le metriche che producono.
:::

::::

```{toctree}
:hidden:
:caption: Panoramica

pipeline
model_kinds
```

```{toctree}
:hidden:
:caption: Dataset

data_sources
featurization
```

```{toctree}
:hidden:
:caption: Modello

architecture
training
aux_supcon
```

```{toctree}
:hidden:
:caption: Fase 2 & orchestrazione

tails
runconfig
```

```{toctree}
:hidden:
:caption: Strumenti & riferimento

tools
examples
```
