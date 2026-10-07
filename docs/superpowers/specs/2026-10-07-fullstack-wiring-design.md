# FullStack: predizione, lettura per stack, cablaggio dei comandi e migrazione (piano B)

Data: 2026-10-07. Stato: **bozza da rivedere** (nessuna modifica al codice).
Seconda metà di `2026-10-06-singlestack-fullstack-design.md`: il piano A (`SingleStack`, `FullStackConfig`, `FullStack` con `fit_body`/`fit_heads`, layout su disco) è implementato su questo branch; qui si fa in modo che tutto il resto del repo usi `FullStack` e si rimuovono i vecchi percorsi supcon.

## Obiettivo

Far funzionare con le run `FullStack` tutto ciò che oggi legge o lancia una run supcon (`predict`/`apply`, `compare`, `benchmark`, `dimred-run/rerun/sweep`, training delle head), poi eliminare i vecchi percorsi di training supcon/supcon_mace e migrare config, script e documentazione.

## Decisioni

- **B1 — Nessuna retrocompatibilità.** Le run prodotte dal vecchio codice (layout piatto supcon, `tails/`, `hierarchical_supcon*`) non si leggono più. Sostituisce D6 del piano A per la parte "lettura legacy". Un solo layout: `stacks/`.
- **B2 — `compare` e `benchmark` lavorano per stack.** Per ogni stack, plot separati e specifici di quello stack; nessuna aggregazione tra stack.
- **B3 — `cgcnn` esce da `compare`/`benchmark`.** Resta allenabile e applicabile con il suo percorso attuale (`dimred-run`, `rerun`, `apply`, `train-tail`), invariato.
- **B4 — `mu2` obbligatorio.** Per `model_kind: supcon` il SOAP si calcola con `soap.element_agnostic: true` (compressione `mu2` di dscribe). Verificato: con `mu2` il vettore è identico (differenza 0.0) cambiando la lista di specie o rinominando gli elementi a geometria uguale.
- **B5 — `dimred-sweep` solo per `FullStack`.** Gli sweep cgcnn non hanno più comando; si ottengono con più `dimred-run`.
- **B6 — Chiavi di history invariate.** `train_family_supcon`/`val_family_supcon` restano nei CSV (verificate dai golden, D11 vieta di toccare le routine di training); `compare` etichetta i grafici dal ruolo dello stack.

## Problemi da evitare (verificati sul codice del branch)

1. **Device.** `supcon/stack.py:106` carica i pesi con `jax.devices(device)[0]` usando `body_train.device` salvato: su una macchina senza GPU, una run allenata su GPU non si carica. La riga è voluta (su GPU i matmul usano TF32 e danno differenze ~1e-3), quindi serve un override **esplicito**: `load(..., device="cpu")`; `predict` ignora `train.device`.
2. **Chiavi `*_family_supcon`.** Negli esperti indicano in realtà la SupCon sullo spacegroup. Vedi B6.
3. **Directory `.*.tmp`.** `FullStack` scrive in `.{nome}.tmp` (sotto `stacks/` e `heads/`) e poi sposta; una run interrotta le lascia. Ogni lettore le ignora.
4. **YAML stretto.** Il parser rifiuta le chiavi sconosciute; i vecchi YAML non si caricano più. Il nuovo schema usa `model_kind`, i vecchi `model:`. Ogni YAML migrato è verificato da un test (sezione 5).

## Sezione 1 — Cosa fa il B

Vedi Obiettivo. Nodo di design: una run vecchia era un modello con un solo spazio latente; una run nuova ne ha fino a 8. `compare`/`benchmark` assumevano `config.yaml`, `loss_history.csv` ed `embeddings.npz` alla radice della run: con il nuovo layout vedrebbero run vuote e le salterebbero senza errori. Da qui B2.

## Sezione 2 — Lettore unico per stack (`pipeline/run_layout.py`)

`open_run(run_dir)` restituisce una vista con `stacks: {nome → StackView}`; ogni `StackView` espone config risolta, `loss_history`, `embeddings.npz`, predizioni e embedding viz delle head, file dei pesi. Un solo layout (`stacks/<nome>/`, `heads/<nome>/`); le directory `.*.tmp` sono ignorate. Nessuna traduzione di nomi vecchi.

- **`compare`:** la suite di plot gira **una volta per nome di stack**, sulle run che hanno quello stack, e scrive in `comparison/<stack>/`. Le run senza quello stack sono saltate con una riga di log. Le etichette delle loss vengono dal ruolo dello stack (B6).
- **`benchmark`:** una riga per coppia (run, stack), colonna `stack`; nessuna aggregazione tra stack né scelta automatica di una "migliore" (metriche come colonne).
- `cgcnn` non passa dal lettore (B3).

## Sezione 3 — Predizione e inferenza

- **Featurizzazione estratta** da `inference.encode_structures` in una funzione indipendente dal tipo di run (SOAP `mu2`, oppure MACE, più standardizzazione). Usata da `predict`. Il ramo `cgcnn` di `encode_structures` resta com'è.
- **`FullStack.predict(structures)`:** feature grezze calcolate **una volta** per struttura (una per combinazione distinta di parametri SOAP, se gli stack differiscono); la **standardizzazione resta per stack** (`feature_mean/std` di ciascuno). Family → sistema più probabile → esperto di quel sistema → spacegroup. Senza esperto per quel sistema, spacegroup mancante esplicito (D9 del piano A), mai un'ipotesi. Senza stack `family`, si usa `predict_stack(nome, strutture)`.
- **Output** per struttura: sistema e probabilità, esperto usato, spacegroup e probabilità, coordinate viz di family ed esperto.
- **Device:** `load(..., device="cpu")` esplicito (problema 1).
- **`mu2` obbligatorio** nel parser di `FullStackConfig` (B4): una config con `soap.element_agnostic: false` è rifiutata prima di costruire qualunque dataset, con un messaggio che indica la correzione. È un amendment al Task 2 del piano A, con test.
- `dimred-apply` su una run nuova usa `predict`; non ha più il ramo supcon.

## Sezione 4 — Comandi ed eliminazioni

- **`dimred-run <config>`:** guarda la chiave del YAML. `model_kind` → `load_full_stack_config`, `FullStack.create`, `fit_body` e `fit_heads` su tutti gli stack elencati; `model:` (cgcnn) → `run_single` come oggi.
- **`dimred-rerun`:** rilegge `config.yaml` della run e la rilancia in una cartella nuova.
- **`dimred-sweep`:** espande percorsi puntati sul YAML completo (`family.encoder.latent_dim`, `experts.cubic.train.epochs`) e lancia un `FullStack` per combinazione. La regola "salva le feature solo per il primo run con lo stesso dataset" passa a livello di stack.
- **`dimred-train-heads <config> <run_dir> [--stacks a,b]`** (nuovo): allena un set di head su corpi già salvati. Sostituisce `dimred-train-tail` per supcon; `dimred-train-tail` resta per cgcnn.
- **`dimred-compare`, `dimred-benchmark`:** solo `FullStack` (sezione 2).

**Si cancella** (dopo rilettura dei rami condivisi con cgcnn, un ramo alla volta): i rami supcon/supcon_mace di `run_single` e l'auto-tail `RunConfig.tails`; `_train_hierarchical_supcon` e i rami supcon di `train_tail`; `HierarchicalSupconTailConfig` con i campi `sg_*`. Prima della cancellazione `_classifier_eval_plots` si sposta in un modulo condiviso.

## Sezione 5 — Migrazione e verifica

- **YAML (29 in `configs/`):** si migra un insieme piccolo e rappresentativo (run singola, run pyxtal, `best_combo`, `hierarchical` come esempio con esperti, uno sweep, un `train_heads`). Le config cgcnn restano, tranne `sweep_cgcnn`, `tuning_sweep_cgcnn` e `slurm/tuning_sweep_cgcnn.yaml`, che si eliminano. Si eliminano anche i `tail_train_*` di `round19_sweep/` e gli esempi `supcon_mace` smoke/scale (la git history li conserva).
- **Script/notebook:** `evaluate_holdout_pyxtal.py`, `evaluate_ns_trajectories.py` e `ns_grid_from_trajectories.ipynb` passano a `FullStack.open(...).predict`. `tune_sg_visualization_hidden_dims.py` (le dimensioni viz per famiglia sono ora un override per esperto, D8) si rilegge nel piano e si riscrive con `fit_heads` o si elimina.
- **`slurm/` (8 script):** aggiornati comandi e percorsi; il blocco di retry sui nodi `.novalocal` non si tocca.
- **Docs:** le 8 pagine in `docs/*.md` e i `CLAUDE.md` (radice, `supcon/`, `pipeline/`) riscritti corti, con la trappola delle chiavi `*_family_supcon`; `sphinx-build -W` in chiusura.
- **Test config:** `test_example_configs_load` percorre `configs/**/*.yaml` e verifica ciascun file con il suo loader (`load_full_stack_config` per `model_kind`, loader delle head per `train_heads_*`, loader attuale per cgcnn).
- **Altri test nuovi:** `predict` (instradamento, esperto mancante, feature una volta con `mu2`); lettore che ignora `.*.tmp`; `load(device="cpu")` con `device: gpu` salvato; `mu2` obbligatorio; smistamento di `dimred-run` per chiave; `compare` e `benchmark` per stack su una mini-run. I test dei percorsi rimossi (`test_pipeline_single_run`, `_tail_training`, `_tail_config`, `_cli`, `_sweep`, `_compare`, `_benchmark`, `_inference`, `_config`, `_dataset_cache`, `test_supcon_tail_training`) sono riscritti o eliminati.
- Solo i test dei file toccati, mai la suite intera di iniziativa.

## Sezione 6 — Ordine, rischi, fuori perimetro

Ordine (un commit per passo, su `feature/singlestack-fullstack`):

1. `mu2` obbligatorio nel parser di `FullStackConfig`, con test.
2. Featurizzazione estratta; `load(device=...)`.
3. `FullStack.predict` e `predict_stack`.
4. `run_layout`; `compare` e `benchmark` per stack.
5. Comandi: `dimred-run` (smistamento), `rerun`, `sweep`, `dimred-train-heads`, `apply`.
6. Spostamento di `_classifier_eval_plots`; cancellazione dei vecchi percorsi supcon.
7. Migrazione di YAML, `slurm/`, script, notebook; `test_example_configs_load`.
8. Docs, `CLAUDE.md`, `REFACTOR_TODO.md`, `sphinx-build -W`.

I passi 1–4 aggiungono codice senza rompere nulla; 5–6 cancellano e si fanno a 1–4 verdi.

**Rischi:** rami condivisi con cgcnn dentro `run_single`/`train_tail` (152 riferimenti a `model_kind` nei file toccati) → si rilegge e si cancella un ramo alla volta; config marce scoperte tardi → test sui YAML; mezze scritture da run interrotte → lettore che ignora `.*.tmp`. I punti minori del ledger del piano A sono andati persi e non sono ricostruibili.

**Fuori perimetro:** run vecchie, `cgcnn` nello stack uniforme, Materials Project in `FullStack`, unificazione dei cicli di training e di `ProjectionTail`/`VisualizationTail`, `slurm/_common.sh`, spezzare i file grandi.
