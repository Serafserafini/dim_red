# Confronto tra model_kind

`supcon` e `supcon_mace` (chiave YAML `model_kind:`) si allenano solo con `FullStack` ({doc}`fullstack`); `cgcnn` (`model: cgcnn`, `RunConfig`) è l'unico percorso di `run_single`. Il `model_kind` sceglie l'intera famiglia di modello. Ogni scheda descrive **cosa fa**, non un numero misurato — vedi {doc}`tools` per confrontare run reali.

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
: Per stack: body (corpo + proiezione), poi heads (classifier + viz)

Config chiave
: per stack (`FullStack`): `data.soap`, `encoder`, `projection`, `contrastive`, `classifier`, `viz`, `train`, `batching`
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
: come `supcon`, con `data.mace` (richiede `checkpoint_path`) al posto di `data.soap`
:::

:::{grid-item-card} {bdg-success}`cgcnn`
Featurizzazione
: Grafo cristallino (no SOAP)

Obiettivo
: Cross-entropy su family/spacegroup — **richiede** `aux_heads.mode != "none"`

Decoder
: No — solo encoder

Fasi
: Una sola (tail fase 2 con `dimred-train-tail`)

Config chiave
: `graph`, `encoder.latent_dim`, `aux_heads`, `train`
:::

::::

## Come si relazionano

- **supcon** non ricostruisce nulla: impara una rappresentazione separando famiglia/spacegroup via una loss contrastiva supervisionata (Khosla et al. 2020), calcolata su una `ProjectionTail` separata dal corpo. Ogni stack allena poi, a corpo congelato, classificatore e visualizzatore (le *heads*) — vedi {doc}`fullstack`.
- **supcon_mace** è `supcon` con gli embedding di un modello MACE (Batatia et al. 2022, via `mace_jax`) al posto del SOAP: stesso codice di training, stesso `FullStack`. Cattura interazioni a 3 corpi/angolari che il SOAP medio non esprime.
- **cgcnn** è l'unico corpo che non usa né SOAP né MACE: costruisce il proprio grafo di legami e allena direttamente la classificazione dentro il corpo (cross-entropy congiunta), quindi richiede sempre almeno una head attiva.

```{note}
I modelli `vae`, `autoencoder` e `mace` (corpo MACE congelato senza training) sono stati rimossi. In un config `RunConfig` ogni `model:` diverso da `cgcnn` solleva un errore esplicito che rimanda allo schema FullStack.
```

Vedi {doc}`architecture` per il blocco `encoder:` condiviso, {doc}`aux_supcon` per `aux_heads`/`batching`, e {doc}`runconfig` per `RunConfig`.
