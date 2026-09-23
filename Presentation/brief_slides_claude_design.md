# Brief slide per Claude Design

File pensato per essere dato in input a Claude Design per costruire il deck. Contenuto
**sintetico** (titoli + bullet, non prosa) — il testo da dire a voce è in `discorso.md`,
allineato slide per slide con questo. 13 slide, ~20 minuti, talk interno/informale.

Tono generale del deck: pulito, poco testo per slide (il parlato porta il contenuto), tono
onesto/proof-of-concept — evitare grafica che comunichi "risultati definitivi" o
"prodotto finito".

---

### Slide 1 — Titolo

- **Titolo**: [da decidere — proposta: "Riconoscere simmetrie cristalline da strutture
  rumorose: un'idea con SupCon"] — oppure più semplice/diretto, a scelta.
- Sottotitolo: nome/data/occasione del talk.
- Nessun bullet — slide di apertura.

### Slide 2 — Motivazione

- Titolo: **Da dove nasce il problema**
- Bullet:
  - Nested Sampling → traiettoria di strutture rappresentative a diverse T
  - Obiettivo: usarle per costruire il phase diagram, con un'etichetta di fase
  - Caso guida: il brass (ottone), fasi α / β / γ
- Visual: schema concettuale nested sampling → traiettoria → phase diagram con regioni
  etichettate. **Non esiste come figura pronta** — da disegnare ex novo (semplice, no dati).

### Slide 3 — Il problema

- Titolo: **Perché non basta un identificatore di simmetria classico**
- Bullet:
  - Strutture rumorose (mosse MC a T finita): atomi spostati, vacanze
  - Librerie classiche (es. spglib) non funzionano su input rumorosi
  - Ottimizzazione energetica locale: si ferma al minimo locale più vicino
  - Stesso problema con MD ad alte temperature
- Visual: nessuna figura necessaria — slide di solo testo, o un'icona/illustrazione
  concettuale di "struttura rumorosa vs struttura pulita".

### Slide 4 — Lavoro precedente

- Titolo: **Primo tentativo: RDF/Steinhardt + UMAP**
- Bullet:
  - Analisi diretta via RDF / parametri di Steinhardt
  - Oppure: strutture rilassate (MD annealed) → confronto via mappa 2D UMAP
  - Cluster trovati = regioni di T/composizione
  - Proiezione delle fasi α/β/γ di riferimento (brass) → coincidono con i cluster
- Visual: la mappa UMAP con i cluster + proiezione α/β/γ. **Materiale del lavoro
  precedente, fuori da questo repo** — va recuperato da dove è conservato.

### Slide 5 — La domanda

- Titolo: **E se non avessimo strutture di riferimento?**
- Bullet:
  - Serve conoscenza pregressa, indipendente dal sistema chimico
  - Basata sulle simmetrie cristalline in generale, non su un caso particolare
  - Robusta a rumore termico e vacanze
  - → modello ML allenato su strutture *generabili*, non raccolte
- Visual: nessuna, slide di transizione/testo. Eventualmente solo la domanda in grande,
  come frase-chiave della slide.

### Slide 6 — Intro tecnica: descrittore + visualizzazione

- Titolo: **Un descrittore per struttura, esplorabile visivamente**
- Bullet:
  - Caso più semplice: sistemi monoatomici
  - SOAP atomici, mediati → descrittore globale (invariante per permutazione/rotazione)
  - Serve poter esplorare visivamente lo spazio a bassa dimensione (zone di transizione,
    "verso quale simmetria tende")
- Visual: nessuna figura reale pronta — eventuale schema concettuale SOAP (atomo →
  ambiente locale → vettore).

### Slide 7 — Encoder + classifier

- Titolo: **Cosa ci serve: encoder + classifier**
- Bullet:
  - Encoder → rappresentazione compatta
  - Classifier → etichetta di simmetria
  - Perché non Autoencoder/VAE: decoder e ricostruzione superflui, ci interessa solo
    separare simmetrie
- Visual: nessuna — slide di testo/transizione.

### Slide 8 — Architettura SupCon

- Titolo: **Architettura: SupCon a 4 blocchi**
- Bullet/diagramma a blocchi (preferibile come diagramma, non elenco):
  - Encoder → corpo principale, fase 1
  - Projection tail → riduce ulteriormente, loss SupCon calcolata qui, fase 1
  - Classification tail → softmax su corpo congelato, fase 2
  - Visualization tail → 2-3D per plot, fase 2
- Visual: **diagramma a blocchi da disegnare** — encoder al centro, frecce verso projection
  tail (fase 1) e verso classification/visualization tail (fase 2), con etichetta
  "fase 1" / "fase 2" chiara. Nessun dato reale, solo schema. Riferimento testuale:
  `Presentation/01_overview_progetto.md`.

### Slide 9 — Risultati: framing + approccio

- Titolo: **Un proof-of-concept, non un sistema finito**
- Bullet:
  - Dataset generato sinteticamente (pyxtal), 7 famiglie cristalline, rumore aggiunto
    artificialmente
  - Primo tentativo (spacegroup diretto) non ha funzionato → troppa varietà
  - Soluzione: due step — classificatore di famiglia, poi esperto di spacegroup per
    famiglia
- Visual: nessuna figura — slide di testo, tono esplicitamente onesto ("lavoro in corso").

### Slide 10 — Risultati: la mappa 2D

- Titolo: **Le strutture si separano per simmetria**
- Bullet: minimi, la figura è il contenuto principale
- Visual: **scegliere UNA di queste due** (non entrambe):
  - `experiments/plots/round16/grid_all_families_BEST.png` — 7 famiglie, mappa spacegroup
    per ciascuna (più ricca, richiede più tempo per spiegare 7 pannelli)
  - `experiments/plots/round15/viz_family_best_combo.png` — singola mappa famiglia (più
    semplice, più adatta se il tempo stringe)

### Slide 11 — Risultati: l'eccezione

- Titolo: **Un'eccezione: la famiglia Cubic**
- Bullet (massimo 2-3, tono cauto):
  - Difficoltà sistematica, in ogni configurazione provata
  - Non dipende dal modello
  - Sospetto: input/training set di quella famiglia — indagine ancora aperta
- Visual: nessuna figura di confusion matrix/analisi qui (per scelta esplicita — vedi
  `struttura_talk.md`, sezione 7). Slide di solo testo, minimale.

### Slide 12 — Futuro: MACE

- Titolo: **Prossimo passo: embedding MACE**
- Bullet:
  - Sostituire SOAP + encoder allenato con embedding di un MACE pre-addestrato
  - Cattura nativamente interazioni angolari/a molti corpi
  - Ipotesi da testare, anche sui casi più difficili — non una soluzione già in mano
  - Nessun run MACE ancora eseguito: prossimo step concreto
- Visual: nessuna figura reale — eventuale icona/schema "SOAP+encoder → MACE frozen".

### Slide 13 — Chiusura

- Titolo: **Grazie**
- Bullet: nessuno, o una riga sola: "Un'idea che funziona in linea di principio — lavoro in
  corso."
- Visual: nessuna, slide di chiusura standard.

---

## Note per chi userà questo file

- Nessuna slide di questo brief presenta tabelle numeriche, costi di training, o dettagli
  round-per-round — coerente con i vincoli del talk (vedi `struttura_talk.md`).
- Le uniche due figure reali usate sono nella slide 10, e sono già su disco (nessun nuovo
  run necessario) — indice completo delle alternative in `04_figure_disponibili.md`.
- Tre diagrammi concettuali restano da disegnare ex novo (slide 2, 8, eventualmente 6/12) —
  nessuno richiede dati, sono schemi a blocchi/icone.
