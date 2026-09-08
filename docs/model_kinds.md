# Confronto tra model_kind

`RunConfig.model_kind` ∈ `{"vae", "autoencoder", "supcon", "cgcnn", "mace"}` (default `"vae"`, chiave YAML top-level `model:`) sceglie l'intera famiglia di modello. Ogni scheda descrive **cosa fa**, non un numero misurato — vedi {doc}`tools` quando ci saranno run reali da confrontare.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {bdg-primary}`vae`
Featurizzazione
: SOAP

Obiettivo
: Ricostruzione + KL (`train.beta`) + head ausiliarie opzionali

Decoder
: Sì

Fasi
: Una sola

Config chiave
: `vae`, `train`, `aux_heads`
:::

:::{grid-item-card} {bdg-info}`autoencoder`
Featurizzazione
: SOAP

Obiettivo
: Ricostruzione deterministica, nessun KL/`beta`

Decoder
: Sì

Fasi
: Una sola

Config chiave
: `vae` (stesso blocco), `train`, `aux_heads`
:::

:::{grid-item-card} {bdg-warning}`supcon`
Featurizzazione
: SOAP

Obiettivo
: Supervised Contrastive su una `ProjectionTail` — nessuna ricostruzione

Decoder
: No — solo encoder

Fasi
: Fase 1 (corpo + proiezione) + tail fase 2 opzionale

Config chiave
: `vae` (solo encoder), `supcon`, `batching`
:::

:::{grid-item-card} {bdg-success}`cgcnn`
Featurizzazione
: Grafo cristallino (no SOAP)

Obiettivo
: Cross-entropy su family/spacegroup — **richiede** `aux_heads.mode != "none"`

Decoder
: No — solo encoder

Fasi
: Una sola (tail fase 2 ammessa per confronto/ablation)

Config chiave
: `graph`, `vae.latent_dim`, `aux_heads`
:::

:::{grid-item-card} {bdg-secondary}`mace`
Featurizzazione
: Grafo + corpo pre-addestrato congelato (no SOAP)

Obiettivo
: Nessuno — non si allena nulla

Decoder
: No — nessuna testa propria

Fasi
: Solo forward pass; classificazione **solo** via `tails`

Config chiave
: `mace`, `tails`
:::

::::

## Perché queste famiglie, non una sola

- **vae / autoencoder** condividono architettura e API (`VAEArchConfig`, `codec.split_encoder_decoder`) al punto da essere intercambiabili in uno sweep cambiando solo `model:` — la differenza reale è se il latente è campionato (KL, `beta`) o deterministico.
- **supcon** non ricostruisce nulla: impara una rappresentazione separando famiglia/spacegroup via una loss contrastiva supervisionata (Khosla et al. 2020), calcolata su una `ProjectionTail` separata dal corpo. Il corpo, una volta congelato, può ricevere uno o più *tail* in fase 2 — vedi {doc}`tails`.
- **cgcnn** è l'unico corpo che non usa SOAP affatto: costruisce il proprio grafo di legami e non ha alcun obiettivo non supervisionato di fallback, quindi richiede sempre almeno una head ausiliaria attiva.
- **mace** è l'unico che non allena nulla: avvolge un modello equivariante (Batatia et al. 2022) già pre-addestrato su un foundation model, tramite `mace_jax`. `tails` è l'unico modo per ottenerne una classificazione.

Vedi {doc}`architecture` per il blocco `vae:` condiviso, {doc}`aux_supcon` per `aux_heads`/`supcon`/`batching`, e la matrice di compatibilità completa in {doc}`runconfig`.
