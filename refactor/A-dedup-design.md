# Refactor A — deduplicazioni a basso rischio

Data: 2026-10-06. Stato: **bozza da rivedere** (nessuna modifica al codice ancora fatta).
Origine: `REFACTOR_TODO.md` §3, §5b. Precede il refactor B (stack unico famiglia/esperti), che ha una specifica a parte.

## Obiettivo

Eliminare le duplicazioni sicure di codice e di costanti **senza cambiare alcun risultato**: stesse chiavi di cache, stessi array, stessi numeri di training, stessi artefatti. Le run dei round passati devono restare riproducibili e leggibili.

## Fuori perimetro (deciso)

- `ProjectionTail` e `VisualizationTail` restano con i propri `project`/`project_with_params`: la ripetizione è accettata per chiarezza di codice e architettura (decisione dell'utente, 2026-10-06).
- Meccanismo unico delle chiavi deprecate: **già fatto** (`_DEPRECATED_KEYS` in `pipeline/config.py`).
- Routine unica di training SupCon, early stopping e `_iter_batches` duplicati, layout artefatti uniforme, inferenza `hierarchical_supcon`: appartengono al refactor B.
- `cgcnn` nello stack uniforme (§2b) e riorganizzazione dei file grandi (§4): refactor C/D.

## Modifiche

### A1. Config di training gemelle

- `supcon.training.TrainConfig` e `supcon.tail_training.TailTrainConfig` hanno gli stessi 11 campi e gli stessi default, entrambe `frozen` (verificato). `TailTrainConfig` diventa un **alias**: `TailTrainConfig = TrainConfig`, importato da `supcon.training`. La docstring unica dice che `tau`/`distance` servono solo alla loss SupCon (fase 1 e viz tail). I ~100 punti che usano `TailTrainConfig` non cambiano.
- `pipeline.config.TailTrainSettings` **resta una classe a sé** (non ha `val_ratio`: la fase 2 riusa lo split della fase 1). Eventuale riuso dei campi in comune senza cambiare lo YAML.
- `cgcnn.training.TrainConfig` non si tocca (campi diversi: `lambda_family`, `lambda_spacegroup`).

### A2. Costanti ripetute

- `_BATCHING_STRATEGIES` (3 copie: `supcon/training.py`, `supcon/tail_training.py`, `pipeline/config.py`) e `_DISTANCE_METRICS` (2 copie).
- La fonte unica è `supcon/training.py`; `supcon/tail_training.py` le importa.
- `pipeline/config.py` **mantiene la propria copia**: deve restare importabile senza jax (regola del `CLAUDE.md`). Un test confronta le due tuple perché non divergano.

### A3. `dataset_cache.py`: una sola sequenza `get_or_build`

Le 6 funzioni `get_or_build_{dataset, pyxtal_dataset, cgcnn_dataset, pyxtal_cgcnn_dataset, mace_dataset, pyxtal_mace_dataset}` seguono la stessa sequenza (chiave → carica → recupera → augmenta → salva strutture → feature → salva cache → ritorna). Differiscono solo per: funzione di chiave, calcolo feature, salvataggio/caricamento degli array, forma del risultato (7 elementi per SOAP e MACE, 5 per il grafo), testo dei log, e per la sorgente (fetch contro pyxtal).

Design:
- `Featurizer` (dataclass interna): nome per i log, funzione di chiave (fetch e pyxtal), calcolo feature, salvataggio, caricamento. Tre istanze: SOAP, grafo, MACE.
- `StructureSource`: produce `(atoms, labels)`, due varianti (fetch, pyxtal).
- `_get_or_build(featurizer, source, cache_dir, augmentation)`: la sequenza, una volta sola.
- Le 6 funzioni pubbliche restano **con le stesse firme e gli stessi valori di ritorno**, come wrapper sottili. I `build_*_for_run`, i test e gli esempi non cambiano.

Vincoli:
- Le funzioni di chiave (`_cache_key`, `_pyxtal_cache_key`, `_graph_cache_key`, `_pyxtal_graph_cache_key`, `_mace_cache_key`, `_pyxtal_mace_cache_key`) **non si modificano**: devono dare le stesse chiavi di oggi, quindi le cache esistenti (incluse le 559 MB di `old_runs/_dataset_cache`) restano valide.
- Messaggi di log invariati.
- Atteso: `dataset_cache.py` da ~1200 a ~700 righe.

### A4. Rename delle chiavi di history della testa di classificazione

- `train_family_ce`/`val_family_ce` → `train_ce`/`val_ce` (CE = cross-entropy; il nome "family" è un residuo della vecchia testa a due teste e nei tail degli esperti è fuorviante: è la CE sui gruppi spaziali).
- Si cambia dove viene prodotta (`supcon/tail_training.py`) e nei lettori (`pipeline/compare.py` ~241-244, `pipeline/single_run.py` ~248).
- I lettori accettano **entrambi i nomi**, così i `loss_history.csv` già salvati restano leggibili; i CSV nuovi usano il nome nuovo.
- **Non** si toccano le chiavi `family_ce`/`spacegroup_ce` di `cgcnn` (significato corretto) né le chiavi `*_supcon`.
- Si esegue per ultimo, dato che ci sono ~31 occorrenze sparse.

## Verifica

1. **Test di equivalenza della cache (prima di toccare il codice).** Per ciascuna delle 6 combinazioni, su un dataset piccolo (pyxtal, poche strutture; per fetch con `MPRester` mockato), con il codice attuale: salvare chiave, nomi dei file di cache, array e liste restituite come riferimento. Dopo il refactor: chiavi e file identici, array identici (`np.array_equal`), stesso ordine, stessi tipi di ritorno. MACE solo con un `MaceEncoder` finto (non richiede `mace_jax`).
2. **Test A2:** le due tuple di `pipeline/config.py` e di `supcon/training.py` sono uguali.
3. **Test A4:** un `loss_history.csv` con le chiavi vecchie e uno con le chiavi nuove vengono letti allo stesso modo da `compare` e `single_run`.
4. **Test esistenti per file toccato:** `test_pipeline_dataset_cache`, `test_supcon_tail_training`, `test_supcon_training`, `test_pipeline_compare`, `test_pipeline_single_run`, `test_pipeline_tail_training`, `test_pipeline_tail_config`, `test_pipeline_config`. Mai la suite intera di iniziativa; nessun rilancio dopo una passata di sola formattazione.
5. **Parità end-to-end** (come il confronto master contro nuovo già fatto): una run supcon con tail di classificazione e visualizzazione, stesso seed, prima e dopo; devono coincidere dataset, history della fase 1, embedding e tail di visualizzazione. Per la classificazione resta valido il confronto numerico, con le sole chiavi di history rinominate.
6. **Build docs** `sphinx-build -b html -W --keep-going docs` dopo la modifica.

## Rischi

- Raggruppare male le funzioni di salvataggio/caricamento (schemi diversi: `_CACHE_ARRAY_KEYS` per SOAP e MACE, uno proprio per il grafo) può rompere la lettura delle cache esistenti → il test di equivalenza viene prima di tutto.
- Il rename delle chiavi di history ha ~31 punti sparsi → ultimo passo, con il test del lettore retrocompatibile.
- L'alias `TailTrainConfig = TrainConfig` cambia il nome della classe nei messaggi di errore/`repr` → trascurabile, da verificare nei test che controllano il testo.

## Ordine di lavoro

1. Test di equivalenza della cache (scritto sul codice attuale).
2. A3 (cache).
3. A1 e A2 (alias e costanti).
4. A4 (rename delle chiavi, per ultimo).
5. Build docs e parità end-to-end.

Un commit per passo.
