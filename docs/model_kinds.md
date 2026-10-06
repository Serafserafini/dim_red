# Confronto tra model_kind

`RunConfig.model_kind` ∈ `{"supcon", "supcon_mace", "cgcnn"}` (default `"supcon"`, chiave YAML top-level `model:`) sceglie l'intera famiglia di modello. Ogni scheda descrive **cosa fa**, non un numero misurato — vedi {doc}`tools` per confrontare run reali.

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} {bdg-warning}`supcon`
Featurizzazione
: SOAP

Obiettivo
: Supervised Contrastive su una `ProjectionTail` — nessuna ricostruzione

Decoder
: No — solo encoder

Fasi
: Fase 1 (corpo + proiezione) + tail fase 2 opzionali

Config chiave
: `encoder`, `supcon`, `batching`, `train`
:::

:::{grid-item-card} {bdg-secondary}`supcon_mace`
Featurizzazione
: Embedding di un modello MACE pre-addestrato e congelato (no SOAP)

Obiettivo
: Lo stesso di `supcon`, allenato sopra gli embedding MACE

Decoder
: No — solo encoder

Fasi
: Come `supcon`

Config chiave
: `mace` (richiede `checkpoint_path`), `encoder`, `supcon`, `batching`, `train`
:::

:::{grid-item-card} {bdg-success}`cgcnn`
Featurizzazione
: Grafo cristallino (no SOAP)

Obiettivo
: Cross-entropy su family/spacegroup — **richiede** `aux_heads.mode != "none"`

Decoder
: No — solo encoder

Fasi
: Una sola (tail fase 2 ammesse)

Config chiave
: `graph`, `encoder.latent_dim`, `aux_heads`, `train`
:::

::::

## Come si relazionano

- **supcon** non ricostruisce nulla: impara una rappresentazione separando famiglia/spacegroup via una loss contrastiva supervisionata (Khosla et al. 2020), calcolata su una `ProjectionTail` separata dal corpo. Il corpo, una volta congelato, può ricevere dei *tail* in fase 2 — vedi {doc}`tails`.
- **supcon_mace** è `supcon` con gli embedding di un modello MACE (Batatia et al. 2022, via `mace_jax`) al posto del SOAP: stesso codice di training, stessi tail. Cattura interazioni a 3 corpi/angolari che il SOAP medio non esprime.
- **cgcnn** è l'unico corpo che non usa né SOAP né MACE: costruisce il proprio grafo di legami e allena direttamente la classificazione dentro il corpo (cross-entropy congiunta), quindi richiede sempre almeno una head attiva.

```{note}
I modelli `vae`, `autoencoder` e `mace` (corpo MACE congelato senza training) sono stati rimossi: `vae:`/`autoencoder`/`mace` come valore di `model:` solleva un errore esplicito.
```

Vedi {doc}`architecture` per il blocco `encoder:` condiviso, {doc}`aux_supcon` per `aux_heads`/`supcon`/`batching`, e la matrice di compatibilità completa in {doc}`runconfig`.
