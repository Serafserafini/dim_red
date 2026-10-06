# Refactor B — Stack unico per famiglia ed esperti

Data: 2026-10-06. Stato: **bozza da rivedere** (nessuna modifica al codice; le decisioni D1–D5 sono da confermare).
Origine: `REFACTOR_TODO.md` §1, §2. Segue il refactor A (completato, `refactor/A-dedup-design.md`).

## Obiettivo (deciso dall'utente)

I modelli per sola famiglia cristallina e gli esperti per gruppo spaziale sono *sempre* lo stesso oggetto: **encoder + proiezione + visualizzazione + classificatore**, con **una sola implementazione**. Cambia solo l'etichetta su cui si contrasta (famiglia al livello principale, gruppo spaziale locale negli esperti). `ProjectionTail` e `VisualizationTail` restano classi separate e con nome proprio, per chiarezza (decisione dell'utente).

## Stato attuale

| | Livello famiglia | Esperto (per famiglia) |
|---|---|---|
| Corpo + proiezione | `pipeline/single_run.py:run_single` chiama `supcon.training.training_first_phase`; config `encoder:` + `supcon:` + `batching:` + `train:` | `pipeline/tail_training.py:_train_hierarchical_supcon` chiama la stessa `training_first_phase`; config `sg_encoder_hidden_dim`, `sg_latent_dim`, `sg_tau`, `sg_distance`, `sg_lambda_norm`, `sg_projection_dim`, `sg_projection_hidden_dim` |
| Classificatore | `train_tail` (tail_kind `classification`), config `tails.classification.head_hidden_dim` (default 16) | stessa `train_classification_tail`, config `sg_classifier_hidden_dim` (default 32) |
| Visualizzazione | `train_tail` (tail_kind `visualization`), config `tails.visualization.*` (viz_dim, mode, tau, distance, lambda_*, hidden_dim, batching) | stessa `train_visualization_tail`, config `sg_visualization_*` (+ `sg_visualization_hidden_dim_by_family`) |
| Quando si allena | corpo in fase 1; classificatore e viz **separatamente**, ciascuno col proprio comando (`dimred-train-tail`) o in `tails:` | tutto in sequenza dentro un'unica funzione (~400 righe) |
| Artefatti | `model_params.msgpack`, `projection_params.msgpack`, `embeddings.npz`; `tails/classification/…`, `tails/visualization/…` | `tails/<sub>/family/…` (stadio 1) e `tails/<sub>/sg_experts/<famiglia>/{sg_body_params, sg_projection_params, classifier_tail_params, visualization_tail_params, local_spacegroup_classes.yaml, *_loss_history.csv, plot}` |

Le primitive sono già comuni (`training_first_phase`, `train_classification_tail`, `train_visualization_tail`, `SupConEncoder`, `ProjectionTail`, `ClassificationTail`, `VisualizationTail`). A essere duplicata è l'**orchestrazione**, e la **config** (i campi `sg_*` specchiano a mano quelli del livello famiglia, con default che divergono).

## Vincoli di design

- **Il livello famiglia deve poter allenare i tail separatamente.** Le sweep dei round 15–19 allenano molti tail (classificazione, viz con architetture diverse) su uno stesso corpo già allenato, senza riallenarlo. Lo stack unico quindi si compone di **due passi riusabili**, non di una sola funzione monolitica: allenare il corpo + proiezione, poi allenare le teste (classificatore e viz) su `r` congelato.
- **Nessun cambiamento dei risultati.** Stessi seed (oggi tutte le componenti usano `config.seed`), stessi default effettivi, stessi pesi finali. Prima di toccare la logica si scrive un test di equivalenza vecchio contro nuovo (come per A).
- **Compatibilità con ciò che esiste su disco**: i `tail_config.yaml`/`config.yaml` già salvati, i checkpoint, i layout di `tails/` letti da `benchmark`, `compare`, `inference`, gli script `evaluate_*`, lo script `tune_sg_visualization_hidden_dims.py` e i notebook.
- `pipeline/config.py` resta importabile senza jax.

## Proposta

### B1. Un modulo dello stack, senza dipendenze dalla pipeline

Nuovo `src/dim_red/supcon/stack.py` (solo array e modelli, nessun I/O, nessun plot):

```python
@dataclass(frozen=True)
class StackConfig:               # in supcon/stack.py; specchio in pipeline/config.py
    encoder_hidden_dim: Sequence[int]
    latent_dim: int
    tau: float; distance: str; lambda_norm: float            # corpo + proiezione
    projection_dim: int; projection_hidden_dim: Optional[Sequence[int]]
    classifier_hidden_dim: Union[int, Sequence[int]]
    viz_hidden_dim: Sequence[int]; viz_dim: int
    viz_tau: float; viz_distance: str; viz_lambda_norm: float

def train_body(X_train, X_val, labels, config, train_config) -> (SupConEncoder, ProjectionTail, history)
def train_heads(r_train, r_val, labels, config, train_config) -> (ClassificationTail, VisualizationTail, histories)
def train_stack(...)  # = train_body + encode + train_heads
```

`labels` è un piccolo oggetto con le etichette da usare per il contrasto del corpo, per il classificatore e per la viz (famiglia, gruppo spaziale locale, o entrambe con i rispettivi pesi): è l'**unico parametro che cambia tra i livelli**. Le funzioni di training esistenti non cambiano firma.

- **Livello famiglia:** `run_single` chiama `train_body`; `train_tail` (classification/visualization) chiama le due metà di `train_heads` separatamente, come oggi, così i tail restano allenabili uno alla volta su un corpo già salvato.
- **Esperto:** `_train_hierarchical_supcon` chiama `train_stack` per famiglia al posto delle ~250 righe di blocco duplicato. Restano nell'orchestratore, perché sono specifici degli esperti: il fallback sul gruppo spaziale più frequente (`min_samples_per_expert`), il riuso del training come validazione se la famiglia non ha righe di val, l'instradamento famiglia → esperto e l'assemblaggio di `tail_predictions.npz`.

### B2. Una sola config per lo stack

`StackConfig` (specchio in `pipeline/config.py`, come `PyxtalConfig`/`AugmentationConfig`) è l'unica definizione dei campi. Il livello famiglia la costruisce da `encoder:`, `supcon:`, `tails.classification`, `tails.visualization`; gli esperti la leggono da un blocco `expert:` dentro `hierarchical_supcon:`. I default divergenti si riconciliano (quasi tutti già coincidono: corpo `tau 0.05`/`cosine`, viz `tau 0.1`/`euclidean`; `classifier_hidden_dim` 16 contro 32 resta distinto per livello perché i default attuali sono tarati su `best_combo`, vedi D3).

### B3. Layout degli artefatti e compatibilità (vedi D1, D2)

Il codice di lettura (benchmark, compare, inference, script, notebook) dipende dai percorsi e dai nomi dei file attuali, e il layout non cambia i risultati. Si tiene quindi il layout **così com'è**; lo scrive una sola funzione che prende `(stack, directory, prefisso dei nomi)`, con i nomi legacy (`sg_body_params.msgpack`, `classifier_tail_params.msgpack`, …) come parametro, così il layout si può uniformare dopo con una modifica in un punto.

### B4. Inferenza per gli esperti (REFACTOR_TODO §2)

`pipeline/inference.py` guadagna `predict_hierarchical_supcon(run_dir, atoms, hierarchical_subdir)`, costruita sulle stesse funzioni di caricamento dello stack, e gli script `examples/evaluate_holdout_pyxtal.py` e `examples/evaluate_ns_trajectories.py` e il notebook delle traiettorie la usano al posto del codice scritto a mano. Con un test di confronto: stesso risultato di oggi sulle strutture di test.

### B5. (opzionale, dopo B1–B4) Un solo ciclo di training

`training_first_phase` (~390 righe) e `train_visualization_tail` (~330 righe) ripetono quasi riga per riga la validazione degli argomenti, il batching random/balanced, il ciclo per epoche, l'early stopping e la history; `train_classification_tail` ha lo stesso ciclo con un'altra loss. Parametrizzare per "cosa è allenabile e quale loss": un solo `_run_epochs(...)`. È la parte a più alto rischio numerico (stesso ordine di batch, stessi RNG) e **non è necessaria** agli obiettivi B1–B4: va fatta solo con i test di equivalenza di B1 già in piedi.

## Decisioni da confermare

- **D1 — Layout degli artefatti.** (a, raccomandata) Tenere i percorsi e i nomi attuali, scritti da una sola funzione parametrizzata. (b) Uniformare ora (`body_params.msgpack`, `classifier_params.msgpack`, …): serve un lettore di compatibilità per le run esistenti e va aggiornato ogni consumatore.
- **D2 — YAML degli esperti.** (a, raccomandata) Nuovo blocco annidato `hierarchical_supcon.expert:` con i campi di `StackConfig`; le chiavi `sg_*` già in uso (le config tracciate in `configs/` che le usano e i `tail_config.yaml` salvati nelle run) restano accettate e vengono tradotte, con un solo warning di deprecazione. (b) Rompere le vecchie chiavi.
- **D3 — Stadio 1 di `hierarchical_supcon`.** Oggi riallena un proprio classificatore di famiglia (hidden 32) invece di riusare quello del livello famiglia (`tails/classification`, hidden 16). (a, raccomandata) Lasciarlo com'è in questo refactor: riusare quello esistente cambierebbe i numeri e obbligherebbe ad avere sempre un tail di classificazione. (b) Riusarlo (più pulito, ma cambia i risultati e le dipendenze tra tail).
- **D4 — Dove vive lo stack.** (a, raccomandata) `src/dim_red/supcon/stack.py`, solo array/modelli; la pipeline si occupa di I/O, plot e log. (b) Dentro `pipeline/`.
- **D5 — B5 (ciclo di training unico).** (a, raccomandata) Fuori da B: solo se serve, con una specifica a parte. (b) Dentro B, come ultimo passo.

## Verifica

1. **Equivalenza numerica prima di toccare la logica:** un test e uno script di confronto che, su un dataset pyxtal piccolo con stesso seed, allenano (i) una run `supcon` con tail di classificazione e visualizzazione e (ii) una run con `hierarchical_supcon` (esperti inclusi) col codice attuale; salvano pesi, history, embedding e `tail_predictions.npz` come riferimento. Dopo ogni passo i risultati devono essere identici (`np.array_equal`).
2. **Compatibilità config:** un `tail_config.yaml` salvato con chiavi `sg_*` si carica e produce la stessa `HierarchicalSupconTailConfig`/`StackConfig`.
3. **Compatibilità checkpoint e layout:** un tail `hierarchical_supcon` già salvato su disco (`experiments/…`) viene letto da `predict_hierarchical_supcon` con lo stesso risultato dello script attuale.
4. **Test per file toccato**, uno alla volta (mai la suite intera di iniziativa); `sphinx-build -W` alla fine.

## Rischi

- **Seed e ordine dei batch:** qualsiasi cambio nell'ordine con cui si creano modelli o si consumano RNG cambia i pesi. Mitigazione: l'orchestrazione costruisce modelli e li allena nello stesso ordine e con gli stessi seed di oggi; il test di equivalenza lo verifica byte per byte.
- **Superficie di compatibilità ampia:** molte config e script leggono i nomi attuali. Mitigazione: D1(a) e D2(a).
- **`run_single` e `train_tail` restano grandi** (rispettivamente ~750 e ~400 righe): spezzarli per fase è il refactor D, non questo.

## Fuori perimetro

`cgcnn` nello stack uniforme (refactor C), riorganizzazione dei file grandi e `slurm/_common.sh` (D), rinomina dei file di layout (se D1 = a), ciclo di training unico (B5, se D5 = a).

## Ordine di lavoro proposto

1. Test e script di equivalenza (riferimento salvato dal codice attuale).
2. `supcon/stack.py` + `StackConfig` (con test unitari).
3. Esperti: `_train_hierarchical_supcon` usa `train_stack` (stesso risultato).
4. Livello famiglia: `run_single` e `train_tail` usano `train_body`/`train_heads`.
5. YAML `expert:` con traduzione delle chiavi `sg_*`.
6. `predict_hierarchical_supcon` e aggiornamento di script e notebook.
7. Documentazione, `REFACTOR_TODO.md`, build Sphinx.

Un commit per passo.
