# SingleStack e FullStack: unica strada di training per supcon / supcon_mace

Data: 2026-10-06. Stato: **bozza da rivedere** (nessuna modifica al codice).
Sostituisce l'idea del "refactor B" (scartata dall'utente): qui si parte da zero, senza vincoli di riproducibilità dei run passati.

## Obiettivo

Il modello è composto da due livelli con **la stessa identica architettura**:

- un livello che separa/visualizza/classifica i 7 sistemi cristallini (label = sistema cristallino);
- 7 esperti indipendenti, uno per sistema, che separano/visualizzano/classificano gli spacegroup di quel sistema (label = spacegroup).

Ogni livello ha due sottofasi:

1. encoder + projection tail si allenano insieme con la loss SupCon;
2. con l'encoder congelato, classification tail e visualization tail si allenano sull'output dell'encoder (la projection tail è esclusa).

Differenze ammesse tra gli stack: label, dati di training, numero e larghezza degli hidden layer (e, in generale, i valori di config). Nient'altro.

`SingleStack` e `FullStack` diventano **l'unica strada** per allenare `model_kind: supcon` e `supcon_mace`. `cgcnn` non è toccato.

## Decisioni prese

- **D1** — `SingleStack` ha `fit_body` e `fit_heads` separati: si possono allenare più head diverse sullo stesso corpo già salvato (sweep dei tail).
- **D2** — `FullStack` = 1 stack "family" + fino a 7 esperti, **indipendenti**: ognuno ha il proprio dataset, la propria config, il proprio split e la propria standardizzazione. Nessuna logica di gerarchia nell'allenamento.
- **D3** — I dati di ogni stack sono costruiti in modo indipendente. Vale solo per `data_source: pyxtal` (seed diverso per stack). Materials Project non è supportato da `FullStack`.
- **D4** — Gli esperti si allenano sul vero sistema cristallino (non su quello predetto).
- **D5** — Si può allenare un sottoinsieme degli esperti (anche uno solo; anche senza `family`).
- **D6** — Layout su disco nuovo; i run vecchi restano **leggibili** (solo lettura) da `inference`, `compare`, `benchmark`. Gli YAML vecchi non si caricano più per riallenare.
- **D7** — Nessuna label nel YAML: la decide il ruolo (family → sistema, esperto → spacegroup). `SingleStack` riceve le label come array.
- **D8** — Config degli esperti: blocco `experts.defaults` + override per esperto, risolto in una `StackConfig` completa prima del training.
- **D9** — Il fallback "esperto assente → predici lo spacegroup più frequente" sparisce. Troppo pochi dati o meno di 2 spacegroup in un esperto selezionato = errore esplicito.
- **D10** — Non si garantisce la riproducibilità dei risultati dei round passati (dataset e seed per stack nuovi). Nessun test di equivalenza vecchio-contro-nuovo.
- **D11** — Le tre routine di training (`training_first_phase`, `train_classification_tail`, `train_visualization_tail`) e le classi `ProjectionTail`/`VisualizationTail` **non cambiano** (unificarle è un lavoro a parte).

## Unità

### `SingleStack` — `src/dim_red/supcon/stack.py`

Solo array e modelli; nessun I/O, plot o dipendenza dalla pipeline. Riusa `SupConEncoder`, `ProjectionTail`, `ClassificationTail`, `VisualizationTail`, `training_first_phase`, `train_classification_tail`, `train_visualization_tail`.

- `StackConfig`: encoder (hidden, latent), projection, contrastive (tau, distance, lambda_norm), classifier (hidden), viz (hidden, dim, tau, distance, lambda_norm), train (epochs, batch, lr, early stopping), batching, seed.
- `fit_body(X_train, X_val, y_train, y_val) -> history`: encoder + projection insieme.
- `fit_heads(X_train, X_val, y_train, y_val) -> histories`: calcola `r` con l'encoder congelato (`encode`), allena classificatore e viz. I parametri dell'encoder non cambiano.
- `encode`, `classify`, `visualize`, `save(dir)`, `load(dir)`.
- Non sa se è famiglia o esperto.

### `FullStack` — `src/dim_red/pipeline/full_stack.py`

- Per ogni stack selezionato: costruisce il dataset (cache esistente, vedi sotto), split grouped per `material_id` col proprio seed, standardizzazione del proprio dataset, chiama `SingleStack`, scrive su disco.
- `fit_body(stacks=None)`, `fit_heads(name, stacks=None)`: `stacks=None` = tutti quelli presenti in config. Flag equivalente da CLI.
- Aggiungere stack a una run esistente è permesso; scrivere su una cartella `stacks/<nome>/` già esistente è un errore, non una sovrascrittura.
- `predict(structures)`: il livello `family` assegna il sistema, poi si instrada verso l'esperto di quel sistema se esiste. Senza esperto: nessuna predizione di spacegroup (valore mancante esplicito). Senza `family`: instradamento non disponibile; ogni stack si usa da solo.
- Se uno stack fallisce, la run si ferma indicando quale.

### Dataset

Per ogni stack la stessa logica di oggi, chiamata una volta per stack: `pipeline/dataset_cache.py` (pyxtal, augmentation, SOAP o MACE, standardizzazione, cache), `generate.py`, `augmentation.py`. Il blocco dati di un esperto restringe pyxtal ai gruppi spaziali del suo sistema usando i campi già esistenti di `GenerationConfig` (i dettagli si fissano nel piano, dopo aver riletto `generate.py`). Il vocabolario degli spacegroup è locale al dataset di ciascun esperto.

Conseguenza: i dati di `family` e degli esperti non condividono strutture, quindi non esiste una validation comune per valutare la catena famiglia → esperto; serve un test set separato (`examples/evaluate_holdout_pyxtal.py`).

## Config

```yaml
name: ...
seed: ...
model_kind: supcon          # o supcon_mace
family:                     # opzionale
  data:  {pyxtal: ..., augmentation: ..., soap: ...}
  encoder: {...}  projection: {...}  contrastive: {...}
  classifier: {...}  viz: {...}  train: {...}  batching: {...}
experts:
  defaults: {...}           # opzionale, blocco comune
  cubic:  {data: {pyxtal: ...}, encoder: {...}}   # override
  tetragonal: {...}         # solo gli esperti elencati esistono
```

- Gli esperti ammessi sono i 7 sistemi cristallini; solo quelli elencati vengono creati.
- Lo schema è validato prima di costruire qualsiasi dataset.
- Le sweep espandono percorsi puntati (`family.encoder.latent_dim`, `experts.cubic.train.epochs`).
- Le dataclass di config dei dati/encoder/train già in `pipeline/config.py` sono riusate dai nuovi blocchi.
- `pipeline/config.py` resta importabile senza jax.

## Layout su disco

```
<run_dir>/
  config.yaml
  run.log
  stacks/
    <family|cubic|...>/
      config.yaml            # StackConfig risolta
      dataset.extxyz
      body/                  # encoder_params, projection_params, loss_history.csv
      embeddings.npz         # embeddings, labels, material_ids, split, feature_mean/std
      heads/<nome>/          # classifier_params, viz_params, loss history,
                             # predictions.npz, plot
```

Una sola funzione di scrittura per tutti gli stack. `heads/<nome>/` sostituisce `tails/<kind>/`.

### Lettura legacy

`inference`, `compare`, `benchmark` passano da un lettore che riconosce il layout: `stacks/` presente = nuovo, altrimenti il vecchio (`model_params.msgpack` alla radice + `tails/...`, incluso `tails/hierarchical_supcon*`). Restituisce la stessa vista ai due casi. Solo lettura.

## Cosa si riusa e cosa si rimuove

**Riusato senza modifiche:** primitive di training e modelli di `supcon/`, `dataset_cache`, `generate`, `augmentation`, `soap`, MACE, early stopping, batching, `_split_indices_grouped`, `_build_vocab_ids`, i plot di valutazione (`_classifier_eval_plots`, da spostare in un modulo condiviso se `tail_training.py` perde le parti di training).

**Nuovo:** `SingleStack`, `FullStack`, `FullStackConfig`, writer del layout, lettore legacy.

**Rimosso (dopo aver riletto il codice, per non cancellare logica usata da `cgcnn`):** percorsi di training supcon/supcon_mace in `run_single`, percorso supcon in `train_tail`, `_train_hierarchical_supcon`, `HierarchicalSupconTailConfig` e i campi `sg_*`. I rami `cgcnn` e i check `("supcon","supcon_mace")` collegati restano finché servono.

## Impatto sui consumatori

`dimred-run`, `dimred-sweep`, `dimred-train-tail` chiamano `FullStack`; `benchmark`, `compare`, `inference` leggono tramite il lettore legacy/nuovo; gli script `examples/evaluate_holdout_pyxtal.py`, `evaluate_ns_trajectories.py`, `tune_sg_visualization_hidden_dims.py`, il notebook `ns_grid_from_trajectories.ipynb` e circa 25 YAML in `configs/` (incluso `round19_sweep/`) vanno migrati o eliminati.

## Test

Solo quelli legati ai file toccati, mai la suite intera di iniziativa.

- `SingleStack` su array piccoli: `fit_body` riduce la loss; `fit_heads` non cambia i parametri dell'encoder; `save`/`load` ridanno gli stessi output; a pari seed `fit_body` coincide con una chiamata diretta a `training_first_phase`.
- `FullStack` su pyxtal minuscolo: dataset distinti per stack, layout su disco, `fit_heads` ripetuto sullo stesso corpo, sottoinsieme di esperti, errore su cartella già esistente, errore su dati insufficienti, `predict` con esperto mancante.
- Config: merge `defaults` + override, validazione, espansione dei percorsi nelle sweep.
- Lettore legacy: fixture del vecchio layout letta da `inference`, `compare`, `benchmark`.
- Test dei percorsi rimossi: eliminati o riscritti.

## Ordine di lavoro (un commit per passo)

1. `SingleStack` + test.
2. `FullStackConfig`, `FullStack`, layout su disco + test.
3. Lettore legacy in `inference`, `compare`, `benchmark` + test.
4. Collegamento di `dimred-run`/`sweep`/`train-tail` a `FullStack`; rimozione dei vecchi percorsi di training supcon.
5. Migrazione di config d'esempio, script, notebook, `CLAUDE.md`, `REFACTOR_TODO.md`.

## Rischi

- **Superficie ampia** (~25 YAML, script, notebook, ~15 file di test): il passo 5 è lungo ma meccanico.
- **Logica non emersa nella mappatura** dentro `run_single`/`train_tail` (rami condivisi con `cgcnn`): si rilegge prima di rimuovere.
- **Dataset indipendenti**: costo di generazione fino a 8× per una run completa (mitigato dalla cache per config identiche).
- **Niente confronto con i round passati** a parità di config (D10): le metriche nuove vanno confrontate solo tra run nuovi.

## Fuori perimetro

`cgcnn` nello stack uniforme, unificazione dei tre cicli di training e di `ProjectionTail`/`VisualizationTail`, Materials Project in `FullStack`, riuso del classificatore di famiglia negli esperti, spezzare i file grandi.
