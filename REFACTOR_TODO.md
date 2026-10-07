# Refactor futuro (lista di cose da fare)

Refactor A (dedup sicure) completato: vedi `refactor/A-dedup-design.md` e `refactor/A-dedup-plan.md`.
Note di lavoro nate dalla rassegna funzione-per-funzione del 2026-10-01/05.
**Non è un piano di pulizia**: la pulizia (eliminare codice morto/superato) si fa prima, separatamente. Questi punti sono da affrontare *dopo*, con i test al verde, uno per volta.

Stato delle voci: **[deciso]** = l'utente ha già indicato l'obiettivo; **[ipotesi]** = osservazione mia, da discutere.

---

## 1. [fatto] Un solo "stack" per famiglia ed esperti

Fatto da `SingleStack` (`supcon/stack.py`) + `FullStack` (`pipeline/full_stack.py`): famiglia ed esperti sono lo stesso oggetto (encoder + proiezione + classificatore + viz), un'unica routine di orchestrazione, un'unica config (`StackConfig`), un unico layout su disco (`pipeline/run_layout.py`), valutazione condivisa. Spec: `docs/superpowers/specs/2026-10-06-singlestack-fullstack-design.md`; guida: `docs/fullstack.md`.

Restano aperte le unificazioni *sotto* lo stack (le tre routine di training e le due tail sono rimaste invariate per scelta, D11 della spec):

- [ ] **`ProjectionTail` e `VisualizationTail` = stessa cosa** (principio dell'utente, 2026-10-06): pezzi di modello separati con nome proprio, ma implementazione condivisa (base comune; `VisualizationTail` aggiunge solo il vincolo `output_dim ∈ {2, 3}`). L'unica differenza di comportamento è che la proiezione si allena **insieme all'encoder** (fase 1) mentre la viz tail si allena **dopo, su `r` congelato** (fase 2). Oggi `__init__`, `project`, `project_with_params` sono duplicati in `supcon/tails.py`.
- [ ] **Una sola routine di training SupCon** per `training_first_phase` e `train_visualization_tail` (copie quasi riga per riga: validazione argomenti, batching random/balanced, ciclo per epoche, early stopping, history). Parametrizzare per "cosa è allenabile": corpo + proiezione insieme, oppure solo la tail su rappresentazioni congelate.

**Cautela:** scrivere prima un test di equivalenza (stessi input, stessi pesi finali); i riferimenti golden di `SingleStack` sono in `tests/golden/`.

## 2. [fatto] Inferenza con gli esperti

Fatto da `FullStack.open(run).predict(...)` (famiglia assegna il sistema, poi l'esperto di quel sistema; esperto mancante -> `None`), `pipeline/featurize.py` e `dimred-apply`; `examples/evaluate_holdout_pyxtal.py` e `examples/evaluate_ns_trajectories.py` usano quella.

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

`pipeline/config.py` (~1060 righe), `pipeline/compare.py` (~1020), `pipeline/dataset_cache.py` (~1010), `supcon/training.py` (~770), `supcon/tail_training.py` (~660).
- [ ] Spezzare per responsabilità (es. `config.py` in `config/{run,tails,model}.py`), mantenendo gli import pubblici.

## 5. [ipotesi] Config e compatibilità

- [ ] Le dataclass gemelle `pipeline.config.PyxtalConfig` / `AugmentationConfig` specchiano a mano quelle di `dim_red.generate` / `dim_red.augmentation` (voluto, per non importare mp_api/pymatgen). Valutare se un'unica sorgente può generare l'altra.

## 5b. [ipotesi] Script slurm

- Ogni `.sbatch` ripete ~20 righe identiche (retry su nodi `.novalocal` + attivazione dell'ambiente conda `dmred` + controllo `CONDA_DEFAULT_ENV`). Estrarle in un unico `slurm/_common.sh` incluso con `source`, così un fix (es. la lista dei nodi da escludere) si fa in un punto solo.

## 6. [ipotesi] Documentazione

- [x] `CLAUDE.md` (root, `pipeline/`, `supcon/`, `mace/`) rigenerati corti dopo FullStack; `README.md` riscritto.
