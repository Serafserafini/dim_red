# Figure reali già disponibili, per sezione del talk

Nessuna di queste richiede un nuovo run — sono tutte già su disco. Percorsi relativi alla
root del repo (`/home/seraftu/Work/code/dim_red/`). Indice più dettagliato e ragionato,
round per round: `experiments/plots/README.md` (lì ci sono anche le spiegazioni di cosa
mostra ogni variante `_BEST` e perché è stata scelta).

## Sezione 7 — Risultati

### Dataset / augmentation
Nessun plot dedicato pronto (istogramma composizione dataset, conteggio strutture per
famiglia). Da generare se serve, non presente su disco — l'unico riferimento simile nel
vecchio `presentation_notes.md` (`spacegroup_histogram.png`) punta a un run che non è più
presente ai percorsi originali.

### Confronto vae / autoencoder / supcon (round 1, identità chimica presente)
- `experiments/milestones/round1_2026-09-10/plots/accuracy_by_model_kind.png`
- `experiments/milestones/round1_2026-09-10/plots/embedding_quality_2d_by_model_kind.png`
- `experiments/milestones/round1_2026-09-10/plots/viz_plot_family_vae.png` /
  `viz_plot_family_autoencoder.png` / `viz_plot_family_supcon.png` /
  `viz_plot_family_euclidean.png` — mappe 2D famiglia, una per model_kind/variante.
- `experiments/milestones/round1_2026-09-10/plots/viz_plot_family_supcon_richtail.png` —
  supcon con la "tail ricca" che chiude il divario con vae (Atto 3 di `storia_studio.md`).

### Corpo di riferimento (round 7, `element_agnostic: true`, family accuracy 0.9726)
- `experiments/plots/round7/confusion_matrix_topmodel_val.png` — matrice di confusione
  famiglia.
- `experiments/plots/round7/viztail_topmodel_family_only.png` — mappa 2D, tail semplice.
- `experiments/plots/round7/viztail_topmodel_family_and_spacegroup_cosine.png` /
  `_euclidean.png` (+ varianti `_spacegroup.png`) — tail ricca, la ricetta standard
  emersa nell'Atto 3.

### Classificatore diretto famiglia→spacegroup (round 13, il migliore in assoluto per SG)
- `experiments/plots/round13/viz_expert_chained_<Famiglia>.png` (7 famiglie) — miglior
  qualità 2D tra i tre design provati in quel round per l'esperto di spacegroup.
- Numero di riferimento (non un plot): oracle spacegroup accuracy **0.9322**
  (`tails/hierarchical_r13_soap_experts_BEST/`).

### Miglior corpo "simmetrico" SupCon-SG (round 15 `best_combo`)
- `experiments/plots/round15/viz_family_best_combo.png` — mappa 2D famiglia.
- `experiments/plots/round15/grid_sg_best_combo.png` — griglia spacegroup, 7 famiglie in
  un'unica figura.

### Visualizzazione spacegroup per famiglia, tutte e 7 insieme, versione leggibile
- `experiments/plots/round16/grid_all_families_BEST.png` — **la figura più adatta a una
  slide "risultato finale" sulla visualizzazione**: tutte le famiglie, ciascuna col
  visualizzatore migliore trovato, palette corretta (niente colori ripetuti oltre 10
  classi, a differenza delle versioni round 13-15).

### Su Cubic (round 17) — materiale di approfondimento, NON per il talk da 20 min

Nota: nel talk attuale (`struttura_talk.md`, sezione 7) Cubic è **solo un accenno cauto**,
senza spiegazione di causa-radice — indicazione esplicita dell'utente, che non si sente
ancora sicuro del finding del round 17 nonostante le analisi fatte. Le figure sotto restano
indicizzate qui per un eventuale approfondimento futuro (per te stesso, o se in futuro il
finding ti convince abbastanza da volerlo raccontare), **non da usare in questa versione
del talk**:
- `experiments/plots/round17/confusion_cubic_best.png` — massa diffusa su tutta la
  matrice, nessuna diagonale.
- `experiments/plots/round17/confusion_tetragonal_best.png` — contrappunto diretto: 68
  classi (il numero più alto), ma diagonale quasi perfetta.
- Le altre 5 (`confusion_hexagonal_best.png`, `_monoclinic_`, `_orthorhombic_`,
  `_triclinic_`, `_trigonal_best.png`) per un pannello completo a 7.
- Tabelle numeriche (gradi di libertà per famiglia, soglia 0/1+ gradi di libertà) — solo
  testo, riportate per intero in `storia_studio.md`, Atto 5.

## Sezione 6 — Architettura (diagramma concettuale)

Nessuna figura reale applicabile — è un diagramma a blocchi (encoder → projection/
classification/visualization tail), da disegnare ex novo. Vedi la tabella testuale in
`struttura_talk.md`, sezione 6, come contenuto di partenza.

## Sezioni 1-4 — Motivazione, problema, lavoro precedente

Fuori da questo repo (materiale del progetto Nested Sampling/brass). Nulla da indicizzare
qui.
