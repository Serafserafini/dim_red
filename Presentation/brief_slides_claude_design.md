# Brief slide per Claude Design

File pensato per essere dato in input a Claude Design per costruire il deck. Contenuto
**sintetico** (titoli + bullet, non prosa) — il testo da dire a voce è in `discorso.md`,
allineato slide per slide con questo. 14 slide, ~23-25 minuti (sopra il target di 20 — vedi
nota sul taglio in cima a `discorso.md`), talk interno/informale.

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
  - Esempio (solo per motivazione/lavoro precedente, non un filo conduttore di tutto il
    talk): il brass (ottone), fasi α / β / γ
- Visual: schema concettuale nested sampling → traiettoria → phase diagram con regioni
  etichettate. **Non esiste come figura pronta** — da disegnare ex novo (semplice, no dati).

### Slide 3 — Il problema

- Titolo: **Perché non basta un identificatore di simmetria classico**
- Bullet:
  - Strutture rumorose (mosse MC a T finita): atomi spostati, vacanze, alcune nemmeno
    vicine a un minimo locale sulla PES
  - Librerie classiche (es. spglib) non funzionano su input rumorosi
  - Ottimizzazione energetica locale: costosa (fattibile solo su una parte delle
    strutture) e comunque non garantisce di arrivare alla fase reale
  - Stesso problema con MD ad alte temperature
- Visual: nessuna figura necessaria — slide di solo testo, o un'icona/illustrazione
  concettuale di "struttura rumorosa vs struttura pulita".

### Slide 4 — Lavoro precedente

- Titolo: **Primo tentativo: SOAP + UMAP**
- Bullet:
  - Strutture rappresentative rilassate (MD annealed) → descrittori SOAP → mappa 2D con
    UMAP
  - Cluster trovati = regioni di T/composizione
  - Proiezione delle fasi α/β/γ di riferimento (brass) → coincidono con i cluster
- Visual: la mappa UMAP con i cluster + proiezione α/β/γ. **Grafico già esistente** — va
  solo recuperato da dove è conservato, non è materiale di questo repo.

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
  - Stesso descrittore già visto in slide 4: SOAP atomici, mediati → descrittore globale
    (invariante per permutazione, rotazione, traslazione, scelta della cella)
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

### Slide 8 — SupCon: la teoria

- Titolo: **Come funziona la loss SupCon**
- Bullet: minimi — la formula e l'esempio portano il contenuto, non un elenco puntato
- Contenuto principale (da mostrare sulla slide, non solo dire a voce):
  - **Formula** (usare la forma leggibile, non necessariamente LaTeX):

    per ogni ancora *i*, con P(i) = positivi (stessa etichetta nel batch), A(i) = tutti gli
    altri nel batch:

    L = media su *i* di **−(1/|P(i)|) · Σ_{p∈P(i)} log( exp(sim(zᵢ,z_p)/τ) / Σ_{a∈A(i)} exp(sim(zᵢ,z_a)/τ) )**

    con **sim(zᵢ,zⱼ) = (zᵢ/‖zᵢ‖) · (zⱼ/‖zⱼ‖)** (similarità coseno — la variante che usiamo
    davvero, non la distanza euclidea diretta)
  - Legenda breve sotto la formula: "positivo = stessa simmetria nel batch, negativo =
    simmetria diversa nel batch, τ = temperatura"
- Esempio (riquadro o icona a parte, non nella formula): batch con 3 strutture cubiche + 2
  esagonali → per una cubica, positivi = le altre 2 cubiche, negativi = le 2 esagonali →
  frecce "avvicina" verso i positivi, "allontana" verso i negativi
- Visual: la formula stessa è il contenuto visivo principale — layout pulito, formula
  centrale, legenda piccola sotto, eventuale mini-schema dell'esempio (3 punti cubici + 2
  esagonali con frecce) a lato. Nessun dato reale, puramente concettuale.

### Slide 9 — Architettura SupCon a 4 blocchi

- Titolo: **Architettura: SupCon a 4 blocchi, in due fasi**
- Bullet/diagramma a blocchi (preferibile come diagramma, non elenco), raggruppato per fase:
  - **Fase 1** (encoder + projection tail allenati insieme, loss SupCon della slide
    precedente): Encoder → corpo principale; Projection tail → riduce ulteriormente,
    la loss è calcolata sul suo spazio, non su quello dell'encoder
  - **Fase 2** (corpo congelato, due tail allenate separatamente): Classification tail →
    softmax; Visualization tail → di nuovo loss SupCon, proiettata a 2-3D per il plot
  - Nota implementativa, piccola ma da includere: tutti e quattro i blocchi sono **MLP**
    (Dense + ReLU, nessuna convoluzione, nessuna rete a grafo), sul vettore SOAP a
    lunghezza fissa — non sulla struttura atomica grezza. Scelta voluta di partenza
    semplice per una fase iniziale del progetto, non un limite definitivo — anticipa il
    collegamento con la slide 13 (MACE come prossimo passo più sofisticato)
- Visual: **diagramma a blocchi da disegnare** — encoder al centro, frecce verso projection
  tail (fase 1) e verso classification/visualization tail (fase 2), con etichetta
  "fase 1" / "fase 2" chiara, e un'etichetta piccola tipo "MLP" su ogni blocco per rendere
  visivamente il punto "sono tutti la stessa cosa semplice". Nessun dato reale, solo
  schema. Riferimento testuale: `Presentation/01_overview_progetto.md`.

### Slide 10 — Risultati: framing + approccio

- Titolo: **Un proof-of-concept, non un sistema finito**
- Bullet:
  - Dataset generato sinteticamente (pyxtal), 7 famiglie cristalline, rumore aggiunto
    artificialmente
  - Primo tentativo (spacegroup diretto) non ha funzionato → troppa varietà
  - Soluzione: due step — classificatore di famiglia, poi esperto di spacegroup per
    famiglia
  - Entrambi gli step riusano gli stessi 4 blocchi della slide precedente (cambia solo
    numero/larghezza degli hidden layer di ciascun MLP)
- Visual: nessuna figura — slide di testo, tono esplicitamente onesto ("lavoro in corso").

### Slide 11 — Risultati: il classificatore di famiglia

- Titolo: **Il classificatore di famiglia: separazione netta**
- Bullet:
  - Mappa 2D (blocco visualization) → separazione per famiglia evidente
  - Confusion matrix del classificatore → forte diagonalità, accuratezza media
    **~97%** [confermare il numero esatto sulla figura scelta — vedi nota in `discorso.md`]
- Visual: **due figure, affiancate**:
  - `experiments/plots/round7/viztail_topmodel_family_only.png` — mappa 2D famiglia
  - `experiments/plots/round7/confusion_matrix_topmodel_val.png` — confusion matrix famiglia

### Slide 12 — Risultati: gli esperti di spacegroup, e l'eccezione Cubic

- Titolo: **Per famiglia, uno spacegroup expert — con un'eccezione**
- Bullet:
  - Ogni famiglia ha il proprio esperto di spacegroup (stesso descrittore SOAP,
    completamente indipendenti tra loro)
  - Separazione ancora evidente per quasi tutte — **tranne Cubic**
  - Stesso pattern nelle confusion matrix: l'unica a dare problemi è Cubic
  - Sospetto (non ancora confermato): l'input, il descrittore SOAP stesso per quella
    famiglia — non il modello. Indagine ancora aperta, tono esplicitamente cauto
- Visual: qui la scelta è più delicata — **molte figure candidate, non tutte insieme**:
  - Mappe 2D per famiglia: `experiments/plots/round16/grid_all_families_BEST.png` (tutte
    e 7 in un'unica immagine — più comodo di 7 slide separate)
  - Confusion matrix per esperto: `experiments/plots/round17/confusion_<famiglia>_best.png`
    (7 file separati) — se si vogliono mostrare, il minimo utile è **due**:
    `confusion_cubic_best.png` (quella con problemi) affiancata a una qualunque delle
    altre 6 come contrasto (es. `confusion_tetragonal_best.png`), non serve mostrarle
    tutte e 7
  - **Se la slide risulta troppo densa in prova**: tenere solo una delle due categorie di
    figure (o solo le mappe 2D, o solo le due confusion matrix di contrasto), non
    entrambe — il discorso in `discorso.md` regge comunque con meno immagini in slide

### Slide 13 — Futuro: MACE

- Titolo: **Prossimo passo: embedding MACE**
- Bullet:
  - Sostituire SOAP + encoder allenato con embedding di un MACE pre-addestrato
  - Cattura nativamente interazioni angolari/a molti corpi
  - Ipotesi da testare, anche sui casi più difficili — non una soluzione già in mano
  - Nessun run MACE ancora eseguito: prossimo step concreto
- Visual: nessuna figura reale — eventuale icona/schema "SOAP+encoder → MACE frozen".

### Slide 14 — Conclusioni

- Titolo: **Conclusioni**
- Bullet (breve recap dell'intero talk, 4 punti):
  - Problema: strutture da Nested Sampling troppo rumorose per gli strumenti classici
  - Approccio: SupCon su SOAP, famiglia → esperto di spacegroup, da strutture generate
  - Funziona per la maggior parte, un'eccezione aperta (Cubic)
  - Prossimo passo: embedding MACE
- Visual: nessuna figura — slide di solo testo, i 4 bullet sono il contenuto. Eventuale
  "Grazie" più piccolo in fondo alla slide, non come titolo.

---

## Note per chi userà questo file

- Nessuna slide di questo brief presenta tabelle numeriche, costi di training, o dettagli
  round-per-round — coerente con i vincoli del talk (vedi `struttura_talk.md`). La formula
  SupCon (slide 8) è un'eccezione voluta: è teoria del metodo, non un risultato sperimentale.
- Le figure reali usate sono nelle slide 11-12 (mappa famiglia, confusion matrix famiglia,
  mappa spacegroup per famiglia, confusion matrix per esperto) — tutte già su disco, nessun
  nuovo run necessario. Indice completo delle alternative in `04_figure_disponibili.md`.
- Diagrammi concettuali da disegnare ex novo: slide 2 (nested sampling→phase diagram),
  slide 8 (schema esempio positivi/negativi), slide 9 (blocchi encoder/tail), ed
  eventualmente 6/13 — nessuno richiede dati, sono schemi/icone.
