# 1. Il progetto: cos'è `dim_red`

Fonti: `CLAUDE.md` (root), `docs/index.md`, `docs/architecture.md`, `docs/model_kinds.md`.

## In una frase

`dim_red` trasforma database di strutture cristalline in rappresentazioni a bassa
dimensionalità, confrontando più famiglie di modelli — variazionale, deterministico,
contrastivo, a grafo, foundation model congelato — su quanto bene separano famiglia
cristallina e spacegroup nello spazio ridotto.

## La pipeline (in ordine)

1. **Sorgente dati** — due alternative intercambiabili, stesso formato di output
   (`List[ase.Atoms]`):
   - `dim_red.fetch` — interroga il Materials Project (`mp_api`), filtra per sistema
     cristallino, converte pymatgen `Structure` → ASE `Atoms`.
   - `dim_red.generate` — genera strutture sintetiche simmetria-valide con `pyxtal`
     (`GenerationConfig`), **questa è la sorgente usata per l'intero studio "family_only"**
     documentato in `02_timeline_studio.md`.
2. **Augmentation (opzionale)** — `dim_red.augmentation`: jitter posizionale (rumore
   termico simulato) e/o rimozione di vacanze, per diversificare il dataset.
3. **Featurizzazione** — due alternative, a seconda del `model_kind`:
   - **SOAP** (`dim_red.soap`) — descrittore numerico a lunghezza fissa per struttura,
     usato da vae/autoencoder/supcon. La sua dimensione dipende da `n_max`/`l_max`/numero
     di specie chimiche — parametro che si è rivelato **decisivo** nello studio (vedi
     round 2-5 in `02_timeline_studio.md`).
   - **Grafo cristallino** — usato da cgcnn/mace, che non passano mai per SOAP.
4. **Modello** (`model_kind`) — vedi tabella sotto.
5. **Valutazione** — accuratezza di classificazione (famiglia e/o spacegroup) e metriche
   di qualità dell'embedding 2D (silhouette, k-means ARI/NMI, k-NN accuracy) —
   `dim_red.analysis.metrics.embedding_quality_metrics`. **Regola del progetto**: questi
   strumenti non scelgono mai automaticamente una configurazione "migliore" — riportano
   numeri per un giudizio umano (vedi `pipeline.benchmark`, `docs/tools.md`).

## I cinque `model_kind` a confronto

| model_kind | Featurizzazione | Obiettivo | Decoder | Fasi | Config chiave |
|---|---|---|---|---|---|
| `vae` | SOAP | Ricostruzione + KL (`train.beta`) + head ausiliarie opzionali | Sì | Una | `vae`, `train`, `aux_heads` |
| `autoencoder` | SOAP | Ricostruzione deterministica, nessun KL | Sì | Una | `vae` (stesso blocco), `train`, `aux_heads` |
| `supcon` | SOAP | Supervised Contrastive (Khosla et al. 2020) su una `ProjectionTail` — nessuna ricostruzione | No, solo encoder | Fase 1 (corpo+proiezione) + tail fase 2 opzionale | `vae` (solo encoder), `supcon`, `batching` |
| `cgcnn` | Grafo cristallino | Cross-entropy su family/spacegroup — richiede `aux_heads` attive | No | Una (tail fase 2 ammessa per confronto) | `graph`, `vae.latent_dim`, `aux_heads` |
| `mace` | Grafo + corpo pre-addestrato congelato | Nessuno — non si allena nulla | No | Solo forward pass; classificazione solo via `tails` | `mace`, `tails` |

Perché queste famiglie e non una sola:
- **vae/autoencoder** condividono architettura e API al punto da essere intercambiabili in
  uno sweep cambiando solo `model:` — la differenza è se il latente è campionato (KL) o
  deterministico.
- **supcon** non ricostruisce nulla: impara una rappresentazione separando famiglia/
  spacegroup via loss contrastiva supervisionata, calcolata su una `ProjectionTail`
  separata dal corpo (dettaglio architetturale che si è rivelato **causalmente importante**
  — vedi la "detective story" in `03_risultati_chiave.md`).
- **cgcnn** è l'unico corpo che non usa SOAP affatto — costruisce il proprio grafo di
  legami e richiede sempre almeno una head ausiliaria.
- **mace** è l'unico che non allena nulla: avvolge un modello equivariante pre-addestrato
  (Batatia et al. 2022, foundation model MACE-MP-0) — `tails` è l'unico modo per ottenerne
  una classificazione.

## Il workflow a due fasi di SupCon (usato per quasi tutto lo studio)

1. **Fase 1 (`run_single`)** — allena corpo (encoder) + `ProjectionTail` sulla loss
   contrastiva supervisionata. Il corpo produce l'embedding "vero" (es. 8-dim nel
   riferimento round 7); la `ProjectionTail` è uno spazio "usa e getta" (es. 128-dim) su
   cui è calcolata la loss.
2. **Fase 2 (`pipeline.tail_training`)** — corpo **congelato**, si allena una o più
   *tail* separate sopra il suo output: `ClassificationTail`, `VisualizationTail`, o
   `hierarchical`/`hierarchical_visualization` (router famiglia + esperti per-famiglia,
   introdotto nel round 13).

## Cosa NON copre questo documento

Numeri di run reali, grafici, configurazioni vincenti — quello è in `02_timeline_studio.md`
e `03_risultati_chiave.md`. Questo file descrive solo **il comportamento del codice**,
indipendente da qualunque esperimento specifico (coerente con lo scopo di `docs/`, scritto
prima che lo studio iniziasse).
