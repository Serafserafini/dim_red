# Struttura del talk (basata sul brainstorm)

Scaletta ricostruita 1:1 dall'indice del brainstorm (`/home/seraftu/Work/Presentations/Ott26/Brainstorm.pdf`).
Le sezioni 1-4 e l'introduzione tecnica (5) riportano fedelmente quanto già scritto nel
brainstorm, solo riorganizzato per slide — **non ho aggiunto contenuto nuovo lì**, quella
parte è farina del tuo sacco e non è nel codice che posso leggere. La sezione **Risultati**
(7) è invece riempita con i numeri reali dello studio `dim_red` (vedi `storia_studio.md`
per il racconto completo e le fonti). La sezione **Futuro** (8) collega la tua idea
("sostituire SOAP+encoder con embedding MACE") allo stato reale del modulo `mace/` nel
codice.

---

## 0. Titolo

Suggerimento di framing, coerente col resto: *"Dal Nested Sampling alla classificazione
automatica di simmetrie cristalline: un modello ML per fasi senza riferimenti noti"* — o
una variante più corta. Da decidere insieme, non è nel brainstorm.

---

## 1. Motivazione — il problema che ha dato il via

- **Nested Sampling** produce una traiettoria con moltissime strutture rappresentative del
  sistema a diverse temperature.
- Obiettivo: usare queste strutture per definire le regioni del **phase diagram** e dare
  loro un'etichetta di fase (esempio guida: **l'ottone/brass**, con le sue fasi α/β/γ).

**Slide-figura suggerita**: schema concettuale nested sampling → traiettoria di strutture →
phase diagram con regioni etichettate. Non esiste nel repo `dim_red` (è a monte, dal
progetto Nested Sampling) — da costruire ex novo o recuperare da materiale esistente del
gruppo.

---

## 2. Il problema — perché non basta un identificatore di simmetria classico

- Le strutture da mosse Monte Carlo a temperatura finita sono **rumorose**: atomi spostati
  rispetto ai punti esatti di simmetria, vacanze atomiche.
- Le librerie/codici classici di identificazione della simmetria cristallina (es. spglib)
  **non funzionano** su input così rumorosi.
- Anche un'**ottimizzazione energetica locale** non risolve il problema: si ferma al minimo
  locale più vicino, che spesso non corrisponde alla fase corretta.
- Stesso discorso per simulazioni di **dinamica molecolare ad alte temperature**.

Questa slide è il "perché serve altro", il gancio narrativo verso la sezione successiva.

---

## 3. Lavoro precedente — RDF/Steinhardt + UMAP, validato sul caso brass

Cosa è stato fatto **prima** di `dim_red`, come primo tentativo:
- Analisi delle strutture direttamente dalle traiettorie tramite **RDF** o altri parametri
  strutturali come i parametri di **Steinhardt**.
- In alternativa: selezione delle strutture più significative per una data coppia
  (temperatura, composizione), sottoposte a rilassamento (simulazioni MD annealed
  inizializzate con quelle strutture), poi confrontate tra loro con una mappa 2D via
  **UMAP**.
- Risultato: la mappa UMAP mostrava **cluster** corrispondenti a regioni ben definite di
  T/composizione. Proiettando in questo stesso spazio le strutture di riferimento α/β/γ
  del brass (letteratura), i cluster trovati **coincidevano** con le regioni di fase note.

Questa è la prova che il tipo di approccio (descrittore strutturale → spazio 2D → cluster)
*funziona* — ma con un limite forte, discusso nella slide successiva.

**Slide-figura suggerita**: la mappa UMAP con i cluster + le strutture di riferimento
α/β/γ proiettate sopra. Materiale del lavoro precedente al progetto `dim_red`, non
recuperabile da questo repo.

---

## 4. La domanda che porta a `dim_red`

> E se non avessi strutture di riferimento con cui confrontare?

Requisiti per il passo successivo:
- Una **conoscenza pregressa** rispetto alla run corrente, indipendente dal sistema
  chimico specifico.
- Basata sulle **simmetrie cristalline** in generale, non su un caso particolare (brass).
- **Robusta** rispetto a rumore termico e vacanze atomiche.

Soluzione proposta: un **modello ML allenato a riconoscere lo spacegroup**, usando
strutture *generabili* (sintetiche, simmetria-nota per costruzione) come dati di training
— non strutture di riferimento raccolte a mano. Questo è esattamente il ruolo che
`dim_red.generate` (pyxtal) gioca nel codice: genera strutture sintetiche simmetria-valide
per ciascuna famiglia/spacegroup, così il modello impara "come appare" ogni simmetria senza
mai aver visto il sistema chimico reale in esame.

---

## 5. Introduzione tecnica

### Il caso più semplice: sistemi monoatomici

- Serve un **descrittore globale** per ogni struttura.
- Strategia più naive: **SOAP** atomici, mediati per ottenere un descrittore globale —
  invarianza per permutazione, rotazione, ecc. (`dim_red.soap`, media "outer" su tutti gli
  atomi della struttura).

### Perché la visualizzazione 2D conta

Quando il sistema si trova in **zone di transizione** o la fase non è ben definita, è utile
poter identificare visivamente "dove si trova" una struttura e verso quale simmetria
"tende" — non solo un'etichetta discreta. Da qui l'esigenza di uno spazio a bassa
dimensione **esplorabile visivamente**, non solo un classificatore black-box.

### Cosa serve, quindi

- **Encoder** — riduce la struttura a una rappresentazione bassa-dimensionale.
- **Classifier** — assegna un'etichetta di simmetria (famiglia/spacegroup) a quella
  rappresentazione.

### Perché non Autoencoder/VAE

Il primo pensiero per un "encoder" va naturalmente ad Autoencoder o VAE. Ma nel nostro
caso la capacità di ricostruzione e l'intero decoder sono **completamente superflui** —
quello che interessa è separare simmetrie diverse tra loro, non ricostruire l'input. Questo
è esattamente il motivo per cui il codice sceglie **SupCon** invece di AE/VAE come
architettura principale (vedi `docs/model_kinds.md`: supcon è l'unico `model_kind` senza
decoder, "no reconstruction at all").

*(Nota per la slide: lo studio ha comunque confrontato empiricamente vae/autoencoder/supcon
a parità di condizioni nel round 1 — tutti e tre arrivano a accuratezze simili su questo
task, ma solo supcon è motivato dal principio "niente ricostruzione, solo separazione". Se
utile come controprova quantitativa di questa scelta di design, vedi `storia_studio.md`,
Atto 1.)*

---

## 6. Architettura scelta: SupCon a 4 blocchi

Riferimento: Khosla et al., *Supervised Contrastive Learning*, NeurIPS 2020 (arXiv:2004.11362)
— l'articolo citato nel brainstorm come ispirazione. La loss SupCon ha come obiettivo
diretto la separazione delle classi in uno spazio a dimensione ridotta.

Quattro blocchi, montabili insieme (`dim_red.supcon`):

| Blocco | Cosa fa | Quando si allena |
|---|---|---|
| **Encoder** | Corpo principale di parametri — la rappresentazione condivisa da tutti gli altri blocchi | Fase 1 |
| **Projection tail** | Riduce ulteriormente l'output dell'encoder; è lo spazio su cui è calcolata la **SupCon loss** | Fase 1, insieme all'encoder |
| **Classification tail** | Prende l'output dell'encoder (corpo **congelato**), softmax classica | Fase 2, dopo aver fissato i pesi del corpo |
| **Visualization tail** | Porta l'output dell'encoder a 2-3 dimensioni, per plottare i cluster di simmetria | Fase 2 |

**Slide-figura suggerita**: diagramma a blocchi encoder → (projection tail / classification
tail / visualization tail), con frecce che indicano quale blocco è attivo in fase 1 vs
fase 2. Diagramma concettuale, nessun dato — coerente con `01_overview_progetto.md`.

Un dettaglio emerso *dallo studio* (non nel brainstorm originale, ma rilevante se la
domanda arriva dal pubblico): perché il corpo non collassa se la loss passa da una
proiezione separata? Vedi Atto 3 di `storia_studio.md` — la storia della "vecchia run"
comparsa a agosto e del commit che ha introdotto la `ProjectionTail`. Buona slide di
backup/domande.

---

## 7. Risultati

Fonte completa con tutti i numeri: `storia_studio.md`. Qui solo lo scheletro allineato ai
punti del brainstorm.

### Generazione del dataset e augmentation

- Dataset generato con `pyxtal` (non Materials Project): 7 famiglie cristalline
  (triclinic, monoclinic, orthorhombic, tetragonal, trigonal, hexagonal, cubic), 500
  strutture "grezze" per famiglia.
- **Augmentation** (`dim_red.augmentation`, config di riferimento — `configs/tuning_sweep_supcon_family_only_round7.example.yaml`):
  `n_augmented: 4` + originale mantenuto (`keep_original: true`) → **5 copie per
  struttura**, per un totale di **17.500 strutture**. Ogni copia augmentata ha, con
  probabilità 0.5 ciascuno, jitter posizionale (rumore Gaussiano, `jitter_std: 0.05 Å`)
  e/o rimozione di un atomo (`vacancy_probability: 0.5`, max 1 vacanza) — garantito che
  almeno uno dei due venga applicato a ogni copia augmentata.
- Featurizzazione: **SOAP** (`r_cut=6.0, n_max=8, l_max=6, sigma=0.5`), con
  `element_agnostic: true` (nessuna informazione sulla specie chimica, solo geometria pura
  — la scelta coerente con l'obiettivo "sistema-indipendente" della sezione 4) → descrittore
  a 252 dimensioni per struttura.

### Fallimento del tentativo diretto, e la soluzione a due step

- Primo tentativo: predire **direttamente** lo spacegroup (`family_and_spacegroup` in un
  solo passo) — **fallito**, troppa varietà (decine di spacegroup possibili).
- Soluzione adottata: **processo a due step**.
  1. Un classificatore di **famiglia cristallina** (7 classi).
  2. Per ogni famiglia, un modello **"esperto"** dedicato alla classificazione dello
     **spacegroup** all'interno di quella famiglia.
  - Entrambi gli stadi partono dallo stesso descrittore SOAP e sono **completamente
    indipendenti** tra loro, così come sono indipendenti gli esperti delle diverse
    famiglie.

Numeri concreti di questa architettura a due stadi (corpo di riferimento, famiglia
0.9726-0.9938 a seconda del setup chimico; esperti diretti su SOAP nativo, non
sull'embedding compresso — decisivo, +32 punti rispetto a partire dall'embedding a 8-dim):
**accuratezza famiglia** fino a 0.97-0.99, **accuratezza spacegroup oracle** fino a
**0.93** con gli esperti meglio configurati. Dettaglio completo: `storia_studio.md`, Atto 4.

### Classification matrix e mappe 2D (famiglia + spacegroup)

Figure reali già pronte, nessun nuovo run necessario — indice completo in
`04_figure_disponibili.md` / `experiments/plots/README.md`. In sintesi:
- Matrici di confusione per famiglia e per spacegroup-per-famiglia: `experiments/plots/round7/confusion_matrix_topmodel_val.png`,
  `experiments/plots/round17/confusion_<famiglia>_best.png` (una per famiglia).
- Mappe 2D famiglia: `experiments/plots/round15/viz_family_best_combo.png` e varianti.
- Mappe 2D spacegroup per famiglia (tutte e 7 insieme): `experiments/plots/round16/grid_all_families_BEST.png`.

### Il problema di Cubic — dal brainstorm come domanda aperta, ora **risolto**

Il brainstorm pone la domanda esattamente così: *"Problema con i cubic: indipendente dal
modello, soap related? Training data related?"* — questa è oggi una domanda **chiusa**,
con una risposta precisa e verificata (round 17, tre misure indipendenti):

- **Non è training-data related**: Cubic non ha classi sbilanciate, non ha meno atomi/
  struttura delle altre famiglie, e la sua confusione è diffusa su 203 coppie diverse di
  spacegroup (non concentrata su poche coppie "difficili" come le coppie enantiomorfe,
  ipotesi inizialmente sospettata e poi esclusa).
- **È SOAP-related, ma nel senso più profondo**: SOAP è un descrittore **locale**
  (ambiente atomico), e Cubic è l'unico sistema cristallino con **zero gradi di libertà**
  nella forma della cella (a=b=c, tutti gli angoli a 90° per definizione — fatto
  cristallografico esatto, non un'approssimazione). Tutte le altre famiglie hanno almeno
  un parametro di forma libero (1 per esagonale/tetragonale/trigonale, fino a 5 per
  triclina).
- **Scoperta chiave — è un effetto soglia, non un gradiente**: ogni famiglia con ≥1 grado
  di libertà arriva a **≥0.989** di accuratezza oracle sullo spacegroup, indipendentemente
  da quanti gradi di libertà ne abbia in più. Solo Cubic (0 gradi di libertà) crolla al
  **0.497** — un gradino netto, non una discesa progressiva.
- **Indipendente dal modello**: confermato allargando encoder, collo di bottiglia,
  classificatore, visualizzatore — nessuna leva di capacità sposta mai l'accuratezza di
  Cubic in modo significativo, in ogni round in cui è stato testato.

**Conclusione da slide**: il tetto ~50% di Cubic è un limite strutturale genuino di un
descrittore locale come SOAP applicato a un sistema senza alcun grado di libertà di forma
— non un bug, non un problema di training, non risolvibile con più capacità o più dati.
Le uniche strade praticabili sono qualitativamente diverse: descrittori a raggio più lungo,
o feature esplicitamente derivate dalla simmetria dello spacegroup. Ottima slide di
chiusura per la sezione Risultati: risponde in modo definitivo a una domanda che il
brainstorm lasciava aperta. Dettaglio completo con le tre misure e le tabelle numeriche:
`storia_studio.md`, Atto 5.

---

## 8. Futuro — sostituire SOAP + encoder con embedding MACE

Idea del brainstorm: rimpiazzare la coppia "SOAP descriptor + encoder allenato" con
l'embedding di un modello **MACE** pre-addestrato.

Stato nel codice `dim_red` (modulo `mace/`, vedi `docs/model_kinds.md` e
`CLAUDE.md`): questo `model_kind` **esiste già** ed è pensato esattamente per questo uso —
un corpo equivariante MACE-MP-0 **congelato, pre-addestrato** (Batatia et al. 2022/2024,
foundation model su ~1.6M strutture da traiettorie di rilassamento Materials Project),
convertito da Torch a `mace_jax`. A differenza di SOAP+encoder:
- **Non si allena nulla nel corpo** — un solo forward pass per ottenere l'embedding.
- Cattura nativamente interazioni **angolari/a molti corpi** (3-body+), non solo
  l'ambiente locale pairwise che limita SOAP — potenzialmente **la strada più diretta per
  affrontare il problema di Cubic**: se il limite è "SOAP è troppo locale", un corpo
  equivariante pre-addestrato su dati reali potrebbe portare un segnale che SOAP
  strutturalmente non può avere. Collegamento diretto, e piuttosto forte, con la sezione 7.
- La classificazione (famiglia/spacegroup) userebbe la stessa infrastruttura di *tail* già
  costruita per supcon (fase 2: `ClassificationTail`/`VisualizationTail`), applicata sopra
  l'embedding MACE congelato invece che sopra il corpo supcon.

**Stato reale ad oggi**: nessun run MACE esiste ancora in `experiments/` — è un passo
successivo, non ancora eseguito, esattamente come lo presenta il brainstorm ("Futuro").
Config di esempio già pronto nel repo: `configs/single_run_mace.example.yaml`.

---

## Cosa manca per rendere questa struttura una presentazione finita

- Le sezioni 1, 3 (materiale/figure del lavoro Nested Sampling/brass precedente a
  `dim_red`) non sono nel repo `dim_red` — vanno recuperate da dove sono conservate.
- Un diagramma a blocchi per la sezione 6 (encoder/projection/classification/visualization
  tail) — concettuale, nessun dato, descritto sopra ma non ancora disegnato.
- Selezione finale delle figure per la sezione 7 tra quelle elencate in
  `04_figure_disponibili.md` (ce ne sono più di quante ne servano per un talk — è una
  scelta editoriale, non tecnica).
