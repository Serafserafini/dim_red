# Featurizzazione

Tre modi di trasformare una struttura in un vettore, mutuamente esclusivi in base a `model_kind`. Per `supcon`/`supcon_mace` i blocchi `soap`/`mace` stanno sotto `data:` di ogni stack ({doc}`fullstack`).

```{list-table}
:header-rows: 1
:widths: 20 30 50

* - Blocco
  - `model_kind`
  - Idea
* - `soap:`
  - `supcon`
  - Descrittore SOAP per struttura (via `dscribe`), `average="outer"` fisso. Con FullStack `element_agnostic: true` (compressione `mu2`) è obbligatorio.
* - `graph:`
  - `cgcnn`
  - Grafo di legami costruito dal codice stesso, specie chimiche rimappate a slot locali — niente identità chimica reale.
* - `mace:`
  - `supcon_mace`
  - Forward pass attraverso un corpo equivariante **pre-addestrato e congelato**; il corpo SupCon allenato sopra ne usa gli embedding.
```

## `soap:` — Smooth Overlap of Atomic Positions

```{eval-rst}
.. autoclass:: dim_red.pipeline.config.SoapConfig
   :members: as_kwargs
```

## `graph:` — grafo cristallino (CGCNN)

Reimplementazione JAX/Flax di Xie & Grossman 2018. Usa `ase.neighborlist` (corretto per periodicità, valido anche oltre metà larghezza della cella) invece di una matrice di distanze a minima immagine. `latent_dim` **non** è un campo di questo blocco: viene letto da `encoder.latent_dim` (ogni altro campo di `encoder:` è ignorato per `cgcnn`).

```{eval-rst}
.. autoclass:: dim_red.pipeline.config.GraphConfig
   :members: resolved_dmax, n_gaussian, graph_kwargs
```

## `mace:` — embedding da un modello equivariante congelato

Cattura interazioni a 3 corpi/angolari nativamente (message passing equivariante di ordine superiore), a differenza del grafo puramente pairwise di CGCNN. Non c'è un campo `latent_dim`: la larghezza dell'embedding è qualunque cosa produca il checkpoint caricato, letta a runtime — non una scelta di config. Vedi `src/dim_red/mace/CLAUDE.md` per la procedura di conversione del checkpoint (fuori da questa codebase, richiede `mace_jax`).

```{eval-rst}
.. autoclass:: dim_red.pipeline.config.MaceConfig
   :members: mace_kwargs
```
