# Struttura del talk (basata sul brainstorm)

Scaletta ricostruita 1:1 dall'indice del brainstorm (`/home/seraftu/Work/Presentations/Ott26/Brainstorm.pdf`).
Le sezioni 1-4 e l'introduzione tecnica (5) riportano fedelmente quanto già scritto nel
brainstorm, solo riorganizzato per slide — **non ho aggiunto contenuto nuovo lì**, quella
parte è farina del tuo sacco e non è nel codice che posso leggere. La sezione **Risultati**
(7) è riempita con materiale reale dello studio `dim_red`, ma **tenuta volutamente
leggera** (vedi vincoli sotto). La sezione **Futuro** (8) collega la tua idea ("sostituire
SOAP+encoder con embedding MACE") allo stato reale del modulo `mace/` nel codice.

## Vincoli del talk (dati esplicitamente da te, guidano tutta questa scaletta)

- **~20 minuti** — talk corto, non un report esaustivo.
- **Niente racconto round-per-round degli esperimenti**, niente tabelle di
  configurazioni/iperparametri.
- **Niente costi di training.**
- **Obiettivo: chiarire l'idea, non vendere risultati** — framing esplicitamente onesto,
  "non c'è ancora nulla di perfettamente funzionante". Proof-of-concept/lavoro in corso,
  non sistema maturo.
- **Su Cubic: solo un accenno.** Niente gradi di libertà della cella, niente "effetto
  soglia", niente tre misure indipendenti — non hai ancora una spiegazione di cui ti
  senti sicuro, nonostante le analisi già fatte. Una frase sola, cauta: difficoltà
  sistematica su quella famiglia, indipendente dal modello, più probabilmente legata
  all'input/al training set. Resta un'osservazione aperta, non un risultato chiuso.

## Budget di tempo indicativo (per tagliare in prova, non vincolante)

| Sezioni | Contenuto | Tempo |
|---|---|---|
| 1-2 | Motivazione (nested sampling/brass) + problema (simmetrie rumorose) | ~4 min |
| 3-4 | Lavoro precedente (RDF/Steinhardt+UMAP) + domanda che porta a `dim_red` | ~3 min |
| 5-6 | Intro tecnica (SOAP, encoder+classifier) + architettura SupCon a 4 blocchi | ~6 min |
| 7 | Risultati — leggeri, l'idea illustrata, non un report | ~5 min |
| 8 | Futuro (MACE) + framing "lavoro in corso" | ~2 min |

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

## 7. Risultati — leggeri, l'idea illustrata, non un report

Obiettivo di questa sezione: mostrare che l'approccio **funziona in linea di principio** e
dire onestamente **a che punto siamo**, non presentare un report esaustivo di esperimenti.
Frase di apertura suggerita, da dire esplicitamente: *"Quello che segue non è un sistema
finito — è un proof-of-concept che mostra che l'idea regge, con dei limiti che stiamo
ancora capendo."* Per chi vuole il dettaglio completo dopo il talk, tutto il resto è in
`storia_studio.md` — ma non serve per queste slide.

### Il dataset: strutture sintetiche generate, non raccolte

Una riga, non una tabella: le strutture di training sono generate sinteticamente con
`pyxtal` (non prese dal Materials Project) — coerente con il requisito della sezione 4
("conoscenza pregressa, indipendente dal sistema chimico specifico"). Migliaia di
strutture per ciascuna delle 7 famiglie cristalline, con rumore/difetti aggiunti
artificialmente (jitter posizionale, vacanze atomiche) per abituare il modello proprio al
tipo di rumore descritto nella sezione 2. Non serve altro dettaglio qui — niente numeri di
`n_augmented`/`jitter_std`, quelli sono da nota a piè di pagina se qualcuno chiede.

### L'idea a due step: prima la famiglia, poi lo spacegroup

- Primo tentativo: predire **direttamente** lo spacegroup in un solo passo — **non ha
  funzionato bene**, troppa varietà (decine di spacegroup possibili per famiglia).
- Idea adottata, più semplice da spiegare che da un unico modello monolitico: **processo
  a due step**.
  1. Un classificatore di **famiglia cristallina** (7 classi).
  2. Per ogni famiglia, un modello **"esperto"** dedicato allo **spacegroup** solo
     all'interno di quella famiglia.
  - I due stadi partono dallo stesso descrittore SOAP e sono indipendenti tra loro, così
    come sono indipendenti gli esperti delle diverse famiglie — un modello semplice
    "spezzato" in pezzi più piccoli e mirati, non un'architettura più complessa.

Un'unica frase di risultato, senza tabelle: **il passo di famiglia funziona bene**, il
passo di spacegroup funziona **bene per la maggior parte delle famiglie**, con
un'eccezione (vedi sotto). Questo basta per dire "l'idea regge" senza scendere in numeri
round-per-round.

### Una mappa, per far vedere invece che raccontare

Una sola figura vale più di qualunque tabella qui: una mappa 2D dove i punti — ogni punto
una struttura — si raggruppano per famiglia/simmetria. Candidata:
`experiments/plots/round16/grid_all_families_BEST.png` (tutte e 7 le famiglie, ciascuna
con la sua mappa spacegroup) oppure una singola mappa famiglia più semplice
(`experiments/plots/round15/viz_family_best_combo.png`) se 20 minuti non lasciano spazio
per spiegare 7 pannelli. Scegline **una**, non entrambe — indice completo delle
alternative in `04_figure_disponibili.md`.

### Cubic: un'osservazione aperta, non una spiegazione

Una famiglia — Cubic — è sistematicamente più difficile da classificare delle altre sei,
in ogni configurazione provata. Da dire con questa cautela, senza andare oltre:

> Cubic ci dà più filo da torcere delle altre famiglie. Non sembra dipendere dal modello
> (cambiando architettura o capacità il problema resta identico) — il sospetto è che
> c'entri più l'input/il training set per quella famiglia che il modello in sé. È una
> cosa che stiamo ancora indagando, non un risultato chiuso.

Niente di più su questa slide — non gradi di libertà, non "effetto soglia", non tre misure
indipendenti. Se in Q&A qualcuno spinge sul "perché", va bene rispondere con la stessa
cautela ("abbiamo alcune ipotesi ma non ci sentiamo ancora di affermarle con sicurezza")
invece di anticipare qui una spiegazione di cui non ti fidi ancora del tutto — il dettaglio
esiste comunque in `storia_studio.md`, Atto 5, se in futuro ti convince abbastanza da
volerlo raccontare.

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
  l'ambiente locale pairwise di SOAP — un candidato interessante da provare anche per
  vedere se aiuta sui casi più difficili incontrati finora (Cubic incluso), senza
  promettere in slide che lo risolverà: è un'ipotesi da testare, non una soluzione già
  in mano.
- La classificazione (famiglia/spacegroup) userebbe la stessa infrastruttura di *tail* già
  costruita per supcon (fase 2: `ClassificationTail`/`VisualizationTail`), applicata sopra
  l'embedding MACE congelato invece che sopra il corpo supcon.

**Stato reale ad oggi**: nessun run MACE esiste ancora in `experiments/` — è un passo
successivo, non ancora eseguito, esattamente come lo presenta il brainstorm ("Futuro").
Config di esempio già pronto nel repo: `configs/single_run_mace.example.yaml`. Buona slide
di chiusura per ribadire il framing di tutto il talk: **lavoro in corso**, questa è la
direzione, non un traguardo già raggiunto.

---

## Cosa manca per rendere questa struttura una presentazione finita

- Le sezioni 1, 3 (materiale/figure del lavoro Nested Sampling/brass precedente a
  `dim_red`) non sono nel repo `dim_red` — vanno recuperate da dove sono conservate.
- Un diagramma a blocchi per la sezione 6 (encoder/projection/classification/visualization
  tail) — concettuale, nessun dato, descritto sopra ma non ancora disegnato.
- Scegliere **una sola** figura per la sezione 7 tra le due candidate proposte lì (mappa
  famiglia semplice vs griglia a 7 pannelli famiglia+spacegroup) — con 20 minuti a
  disposizione non c'è spazio per più di una.
