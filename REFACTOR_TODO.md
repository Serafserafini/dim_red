# Refactor futuro (lista di cose da fare)

Refactor A (dedup sicure) completato: vedi `refactor/A-dedup-design.md` e `refactor/A-dedup-plan.md`.
Note di lavoro nate dalla rassegna funzione-per-funzione del 2026-10-01/05.
**Non è un piano di pulizia**: la pulizia (eliminare codice morto/superato) si fa prima, separatamente. Questi punti sono da affrontare *dopo*, con i test al verde, uno per volta.

Stato delle voci: **[deciso]** = l'utente ha già indicato l'obiettivo; **[ipotesi]** = osservazione mia, da discutere.

---

## 1. [deciso] Un solo "stack" per famiglia ed esperti

**Obiettivo (utente):** i modelli per sola famiglia cristallina e gli esperti per gruppo spaziale devono essere *sempre* lo stesso oggetto: **encoder + proiezione + viz + classificatore**, con una sola implementazione.

**Stato attuale** (dopo la potatura dei tail_kind `hierarchical`, `hierarchical_visualization` e del single-stage con spacegroup):

| Livello | Orchestrazione | Dove |
|---|---|---|
| Famiglia | fase 1 `train_supcon` (encoder+proiezione), poi due tail separate `classification` e `visualization` | `pipeline/single_run.py` + `pipeline/tail_training.py:train_tail` |
| Esperto (per famiglia) | stessa sequenza, riscritta a mano | `pipeline/tail_training.py:_train_hierarchical_supcon` (~480 righe) |

Le primitive sono già comuni (`supcon.training.train_supcon`, `supcon.tail_training.train_classification_tail` / `train_visualization_tail`, `ClassificationTail`, `VisualizationTail`, `ProjectionTail`). A essere duplicata è l'**orchestrazione**.

**Da fare:**
- [ ] Estrarre una routine unica, ad esempio `train_supcon_stack(features, labels, stack_config) -> {encoder, projection, classifier, visualizer, history}`, chiamata sia da `run_single` (livello famiglia) sia dagli esperti.
- [ ] Una sola dataclass di config per lo stack (encoder/latent/tau/distance/proiezione/classificatore/viz) riusata ai due livelli. Oggi gli esperti hanno campi con prefisso `sg_*` in `HierarchicalSupconTailConfig` che specchiano a mano i campi del livello famiglia.
- [ ] Riconciliare i default che oggi divergono tra i due livelli: `sg_distance: cosine` (encoder esperto) contro `euclidean` (viz); le viz per-famiglia (`sg_visualization_hidden_dim_by_family`). Capire quali differenze sono scelte dichiarate e quali accidenti storici.
- [ ] Layout degli artefatti uniforme (oggi `tails/classification`, `tails/visualization` al livello famiglia contro `tails/hierarchical_supcon_*/sg_experts/<famiglia>/` per gli esperti).
- [ ] Plot di valutazione (`_classifier_eval_plots`) condivisi: stessa griglia per famiglia ed esperti (vedi regola: una confusion matrix per esperto, niente legende nelle griglie viz).
- [ ] La etichetta su cui si contrasta resta l'unico vero parametro che cambia tra i livelli (famiglia contro gruppo spaziale locale).

- [ ] **`ProjectionTail` e `VisualizationTail` = stessa cosa** (principio dell'utente, 2026-10-06): sono pezzi di modello separati e con nome proprio per chiarezza, ma l'implementazione è condivisa (base comune; `VisualizationTail` aggiunge solo il vincolo `output_dim ∈ {2, 3}`). L'unica differenza di comportamento è che la proiezione si allena **insieme all'encoder** (fase 1) mentre la viz tail si allena **dopo, su `r` congelato** (fase 2). Oggi `__init__`, `project`, `project_with_params` sono duplicati in `supcon/tails.py`.
- [ ] **Una sola routine di training SupCon** per `training_first_phase` (ex `train_supcon`, ~390 righe), `train_visualization_tail` (~330 righe) — e, se restano, per la fase delle SG experts. Oggi sono copie quasi riga per riga (validazione argomenti, batching random/balanced, ciclo per epoche, early stopping, history). Parametrizzare per "cosa è allenabile": corpo + proiezione insieme, oppure solo la tail su rappresentazioni congelate.

**Cautele:** i risultati dei round passati devono restare riproducibili (stessi seed, stessi default effettivi). Va scritto un test di equivalenza prima di toccare la logica: stesso input, stessi pesi finali (o stessa accuratezza entro tolleranza) vecchio contro nuovo.

## 2. [deciso, conseguenza della potatura] Inferenza con gli esperti

`pipeline/inference.py:predict_hierarchical` legge solo la config e lo `family_expert_status.yaml` del vecchio `tail_kind: hierarchical` (esperti = solo classificatore). Con quel tail_kind rimosso resta senza uso nella libreria, e **non esiste un equivalente per `hierarchical_supcon`**.

- [ ] Verificare come `examples/evaluate_holdout_pyxtal.py` (che importa `predict_hierarchical`) e `examples/evaluate_ns_trajectories.py` valutano oggi gli esperti `hierarchical_supcon`.
- [ ] Scrivere `predict_hierarchical_supcon` (o, meglio, un'inferenza dello stack unificato del punto 1) nella libreria, e far usare quella agli script.

## 2b. [ipotesi] cgcnn nello stack uniforme

`cgcnn` è l'unico modello che allena la classificazione **dentro** il corpo (cross-entropy congiunta su famiglia e gruppo spaziale, `aux_heads`), mentre supcon allena un corpo con proiezione (fase 1) e poi classificatore e viz come tail separate (fase 2). Per rispettare il principio "encoder + proiezione + viz + classificatore" (vedi §1) la versione allineata sarebbe: **corpo a grafo (`CGCNNBodyModule`) allenato con la loss SupCon + `ProjectionTail`**, poi le stesse tail di classificazione e viz usate da supcon. Così `aux_heads`, `apply_family_mask`, `ClassifierHead`, la testa spacegroup e `train_cgcnn` (loss CE) sparirebbero da cgcnn, e resterebbe solo graph + modello a grafo.

## 3. [ipotesi] Duplicazioni volute tra i pacchetti modello

Il `CLAUDE.md` le documenta come intenzionali (pacchetti indipendenti), ma sono molte:
- `apply_family_mask` ×4 (vae, autoencoder, cgcnn; la copia supcon è in eliminazione)
- `ClassifierHead` ×3 (vae, autoencoder, cgcnn; supcon si fonde in `_MLPTail`)
- `Encoder` (adattatore) ×3 dopo l'eliminazione di quello di supcon
- `_make_optimizer` ×5, early stopping ×4, `_iter_batches` / `_weighted_mean` ×3
- [ ] Se `vae`/`autoencoder`/`cgcnn` restano, valutare un piccolo modulo comune (`dim_red/_common/`) per il solo codice davvero identico. Dipende dalla rassegna di quei pacchetti.

- `VAEDatabase` (`vae/database.py`) è il contenitore dati usato anche da supcon/autoencoder/(cgcnn?): spostarlo in un modulo neutro (es. `dim_red/dataset.py`) così `supcon` non dipende da `vae`.

## 4. [ipotesi] File troppo grandi

`pipeline/config.py` (1854 righe), `pipeline/tail_training.py` (1757), `pipeline/dataset_cache.py` (1187), `supcon/training.py` (1202), `pipeline/compare.py` (1083), `pipeline/single_run.py` (1034).
- [ ] Spezzare per responsabilità (es. `config.py` in `config/{run,tails,model}.py`), mantenendo gli import pubblici.
- [ ] `_train_hierarchical_supcon` e i suoi helper in un modulo a parte (diventa superfluo se il punto 1 viene fatto).

## 5. [ipotesi] Config e compatibilità

- [ ] Le dataclass gemelle `pipeline.config.PyxtalConfig` / `AugmentationConfig` specchiano a mano quelle di `dim_red.generate` / `dim_red.augmentation` (voluto, per non importare mp_api/pymatgen). Valutare se un'unica sorgente può generare l'altra.

## 5b. [ipotesi] Script slurm

- Ogni `.sbatch` ripete ~20 righe identiche (retry su nodi `.novalocal` + attivazione dell'ambiente conda `dmred` + controllo `CONDA_DEFAULT_ENV`). Estrarle in un unico `slurm/_common.sh` incluso con `source`, così un fix (es. la lista dei nodi da escludere) si fa in un punto solo.
- Funzioni identiche duplicate in `pipeline/single_run.py` e `pipeline/tail_training.py`: `_make_unique_run_dir`, `_build_vocab_ids`, `_build_family_spacegroup_mask`, `_save_loss_history` → un solo modulo condiviso (modifica a basso rischio, si può fare anche prima del refactor grosso).
- `run_single` (~730 righe) e `train_tail` (~400) sono funzioni monolitiche: spezzare per fase (dataset → split → modello → training → embedding → artefatti → auto-tail).

## 6. [ipotesi] Documentazione

- [ ] I `CLAUDE.md` (root e per pacchetto) sono molto densi e in parte stantii: dopo la potatura rigenerarli corti, con solo ciò che il codice non dice già.
- [ ] `README.md` riscritto (oggi descrive solo `pca.py` e `utils.py`).
