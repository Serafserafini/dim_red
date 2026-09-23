# Prompt per Claude Design — deck PowerPoint editabile

Questo file contiene tutto il necessario per generare la presentazione con Claude Design:
prima una checklist di cosa preparare/caricare, poi il prompt vero e proprio (un unico
blocco copiabile). Il prompt è autosufficiente — include tutto il contenuto di
`discorso.md` e `brief_slides_claude_design.md`, non serve che Claude Design legga altri
file di questa cartella.

---

## 1. Checklist prima di iniziare

Claude Design (a differenza di questa sessione) probabilmente **non ha accesso al
filesystem locale** — le immagini vanno allegate a mano alla conversazione. Recupera questi
file prima di lanciare il prompt (percorsi relativi a `/home/seraftu/Work/code/dim_red/`):

**Da questo repo (già pronte):**
1. `experiments/plots/round7/viztail_topmodel_family_only.png` — mappa 2D famiglia
2. `experiments/plots/round7/confusion_matrix_topmodel_val.png` — confusion matrix famiglia
3. `experiments/plots/round16/grid_all_families_BEST.png` — mappe spacegroup, 7 famiglie
4. `experiments/plots/round17/confusion_cubic_best.png` — confusion matrix Cubic
5. `experiments/plots/round17/confusion_tetragonal_best.png` — confusion matrix di
   contrasto (una qualunque delle altre 6 famiglie va bene ugualmente, vedi nota nel prompt)

**Da recuperare da fuori questo repo (lavoro precedente al progetto `dim_red`):**
6. Il grafico della mappa UMAP con i cluster T/composizione + proiezione delle fasi α/β/γ
   del brass (per la slide 4) — non è in `dim_red`, va preso da dove l'avete salvato voi.

Se qualcuna di queste non è disponibile al momento, va bene lanciare comunque il prompt:
è scritto per gestire esplicitamente le immagini mancanti (vedi istruzioni nel prompt
stesso) invece di bloccarsi o inventare un placeholder silenzioso.

---

## 2. Il prompt (copia da qui in giù)

```
Voglio che tu crei una presentazione PowerPoint completa ed EDITABILE (file .pptx scaricabile,
non un artifact bloccato) per un talk interno di ricerca, in italiano, di circa 18-19 minuti,
14 slide. Il pubblico è un gruppo di ricerca misto ML + scienza dei materiali che conosce già
concetti come Nested Sampling — non serve spiegare le basi da zero.

CONTESTO DEL PROGETTO (per tua comprensione, non necessariamente da mettere tutto in slide):
Il progetto usa il Nested Sampling per esplorare configurazioni di un sistema a diverse
temperature. L'obiettivo finale è classificare automaticamente la simmetria cristallina di
strutture rumorose (da mosse Monte Carlo o dinamica molecolare ad alta T) per costruire diagrammi
di fase, senza dover disporre di strutture di riferimento note. La soluzione proposta è un modello
di deep learning (SupCon, Supervised Contrastive Learning) allenato su descrittori SOAP di
strutture sintetiche generate al computer (pyxtal), con un'architettura a quattro blocchi
(encoder, projection tail, classification tail, visualization tail) e un processo di
classificazione a due step (prima la famiglia cristallina, poi lo spacegroup con un esperto
dedicato per famiglia).

TONO E STILE — IMPORTANTE:
- Il progetto è un proof-of-concept, non un sistema finito. Il tono deve essere onesto,
  esplorativo, mai trionfalistico. Evita grafica o linguaggio che comunichi "risultato
  definitivo" o "prodotto pronto".
- Slide pulite, poco testo (il relatore parla — le slide sono un supporto visivo, non un
  documento da leggere). Preferisci bullet brevi, diagrammi, figure, non paragrafi.
- Design coerente slide per slide (stessa palette, stesso font, stesso stile di titolo),
  minimale e professionale — non serve elaborato, deve solo essere pulito e leggibile a
  distanza in una sala riunioni.
- Metti, per OGNI slide, il testo indicato come "SPEAKER NOTES" qui sotto nelle note del
  relatore di PowerPoint (non in slide) — è il discorso vero e proprio che il relatore userà
  per provare e per la presentazione live.

GESTIONE IMMAGINI:
- Per le slide che richiedono una figura reale (indicate sotto come "FIGURA ALLEGATA: ..."),
  uso le immagini che ti ho allegato in questa conversazione. Se non trovi l'immagine allegata
  corrispondente a una slide, NON inventare un placeholder silenzioso e non saltare la slide:
  lascia un riquadro segnaposto ben visibile con la scritta "[FIGURA MANCANTE: nome file]" in
  modo che sia facile da individuare e sostituire dopo.
- Per le slide che richiedono un diagramma concettuale (indicate come "DIAGRAMMA DA CREARE:
  ..."), disegnalo tu direttamente nella slide con le forme native di PowerPoint (box, frecce,
  testo) — schemi semplici, stile lavagna pulita, non serve fotorealismo.
- La formula della slide 8 deve essere leggibile e ben formattata — usa l'equation editor di
  PowerPoint o una tipografia matematica chiara (apici/pedici veri, non testo in linea), non
  Unicode approssimativo.

STRUTTURA, 14 SLIDE:

---
SLIDE 1 — Titolo
Titolo grande: "Riconoscere simmetrie cristalline da strutture rumorose: un'idea con SupCon"
(o una variante più semplice se preferisci — è un placeholder, sostituibile).
Sottotitolo: nome del relatore / data / occasione del talk (da lasciare come campo editabile,
non inventare un nome).
Nessun altro contenuto — slide di apertura pulita.
SPEAKER NOTES:
"Buongiorno a tutti. Oggi vi parlo di un'idea su cui stiamo lavorando da qualche mese: usare
il machine learning per riconoscere automaticamente la simmetria cristallina di una struttura,
in situazioni dove i metodi classici non ce la fanno. Vi anticipo subito che non è un lavoro
finito — è più un'idea che sta prendendo forma, e oggi voglio condividere con voi a che punto
siamo, compresi i punti che non abbiamo ancora chiariti del tutto."

---
SLIDE 2 — Da dove nasce il problema
Bullet:
- Nested Sampling → traiettoria di strutture rappresentative a diverse temperature
- Obiettivo: usarle per costruire il diagramma di fase, con un'etichetta di fase per regione
- Esempio guida (solo per motivazione, non useremo l'ottone in tutto il talk): il brass
  (ottone), fasi α / β / γ
DIAGRAMMA DA CREARE: schema concettuale semplice, tre caselle in sequenza con frecce —
"Nested Sampling" → "traiettoria di strutture a diverse T" → "phase diagram con regioni
etichettate". Nessun dato reale, solo concetto.
SPEAKER NOTES:
"Il problema da cui siamo partiti nasce dal Nested Sampling — lo conoscete già, quindi solo un
richiamo veloce: è un metodo di campionamento che esplora lo spazio delle configurazioni di un
sistema, e come sottoprodotto ci dà accesso diretto alla funzione di partizione su un ampio
intervallo di temperature. Il punto per noi è che ci restituisce una traiettoria fatta di
moltissime strutture, ciascuna rappresentativa del sistema a una certa temperatura. L'idea è:
possiamo usare queste strutture per costruire il diagramma di fase — cioè per dire 'in questa
regione di temperatura e composizione, il sistema si trova in questa fase, con questa
simmetria'. Un esempio concreto, che ci servirà tra poco per spiegare cosa avevamo già provato
e perché non bastava, è l'ottone, il brass, che ha diverse fasi — alfa, beta, gamma — ciascuna
con una propria simmetria cristallina."

---
SLIDE 3 — Perché non basta un identificatore di simmetria classico
Bullet:
- Strutture rumorose (mosse Monte Carlo a T finita): atomi spostati, possibili vacanze
  atomiche, alcune nemmeno vicine a un minimo locale sulla superficie di energia potenziale
- Le librerie classiche di identificazione della simmetria (es. spglib) non funzionano su
  input così rumorosi
- Un'ottimizzazione energetica locale non risolve il problema: è costosa (fattibile solo su
  una parte delle strutture) e comunque non garantisce di arrivare alla fase reale
- Stesso problema con simulazioni di dinamica molecolare ad alte temperature
Nessuna figura necessaria — slide di solo testo ben impaginato, eventualmente un'icona
semplice "struttura rumorosa vs struttura pulita" se aiuta visivamente.
SPEAKER NOTES:
"Il problema è che queste strutture non sono pulite. Vengono da mosse Monte Carlo a
temperatura finita, quindi sono rumorose: gli atomi sono spostati rispetto ai punti esatti di
simmetria, e potrebbero esserci vacanze atomiche. Nelle traiettorie del Nested Sampling, poi,
compaiono anche strutture che non sono nemmeno vicine a un minimo locale sulla superficie di
energia potenziale. Questo significa che le librerie classiche per identificare la simmetria
cristallina — penso a spglib — semplicemente non funzionano su input così rumorosi.
Si potrebbe pensare di 'ripulire' ogni struttura con un'ottimizzazione energetica locale prima
di analizzarla, ma qui ci sono due problemi. Il primo è di costo: non possiamo permetterci di
ottimizzare tutte le strutture della traiettoria, quindi al massimo possiamo farlo solo su una
parte di esse. Il secondo, più di fondo, è che anche quando lo facciamo, l'ottimizzazione si
ferma al minimo locale più vicino sulla superficie di energia potenziale — e non c'è nessuna
garanzia che quel minimo corrisponda alla fase reale del sistema. Lo stesso discorso vale per
simulazioni di dinamica molecolare ad alta temperatura.
Quindi ci serve qualcos'altro."

---
SLIDE 4 — Primo tentativo: SOAP + UMAP
Bullet:
- Strutture rappresentative rilassate (dinamica molecolare annealed) → descrittori SOAP →
  mappa 2D con UMAP
- I cluster trovati corrispondono a regioni precise di temperatura e composizione
- Proiettando le fasi α/β/γ di riferimento del brass (letteratura) nello stesso spazio,
  cadono esattamente dentro quei cluster
FIGURA ALLEGATA: la mappa UMAP con i cluster e la proiezione delle fasi α/β/γ del brass
(immagine allegata alla conversazione — se manca, riquadro segnaposto "[FIGURA MANCANTE:
mappa UMAP cluster T/composizione + fasi brass]").
SPEAKER NOTES:
"L'unico approccio che ci ha dato risultati concreti è stato questo: abbiamo preso le
strutture più rappresentative per una certa coppia di temperatura e composizione, le abbiamo
sottoposte a una simulazione di dinamica molecolare annealed, e, usando UMAP per confrontare i
descrittori SOAP delle strutture viste durante questa simulazione, abbiamo costruito una mappa
2D.
Questo ha funzionato sorprendentemente bene: nella mappa si formavano dei cluster netti, che
corrispondevano proprio a regioni precise di temperatura e composizione. E quando abbiamo
proiettato nello stesso spazio le strutture di riferimento delle fasi alfa, beta e gamma del
brass, prese dalla letteratura, sono cadute esattamente dentro quei cluster."

---
SLIDE 5 — E se non avessimo strutture di riferimento?
Bullet (questa slide può anche essere solo la domanda in grande, come frase-chiave):
- Serve una conoscenza pregressa, indipendente dal sistema chimico specifico
- Basata sulle simmetrie cristalline in generale, non su un caso particolare
- Robusta a rumore termico e vacanze atomiche
- → un modello ML allenato su strutture generabili al computer, non raccolte a mano
Nessuna figura — slide di transizione/testo.
SPEAKER NOTES:
"Questo però ci ha lasciato con una domanda: e se non avessimo strutture di riferimento con
cui confrontare? Il brass lo conosciamo bene, ma per un sistema nuovo potremmo non avere
nessuna fase alfa-beta-gamma già pronta in letteratura da proiettare.
Quello che ci serve, allora, è qualcosa che porti già con sé una conoscenza delle simmetrie
cristalline in generale — indipendente dal nostro sistema chimico specifico — e che sia
robusto al rumore termico e alle vacanze atomiche. L'idea è allenare un modello di machine
learning a riconoscere lo spacegroup, usando strutture la cui simmetria è nota e sicura."

---
SLIDE 6 — Un descrittore per struttura, esplorabile visivamente
Bullet:
- Caso più semplice: sistemi con un solo tipo di atomo
- Stesso descrittore SOAP già visto (slide 4): atomici, mediati → descrittore globale,
  invariante per permutazione degli atomi, rotazione, traslazione e scelta della cella
  periodica
- Serve poter esplorare visivamente lo spazio a bassa dimensione: in zone di transizione o
  fase non ben definita, aiuta vedere "dove si trova" una struttura e verso quale simmetria
  tende
Nessuna figura reale pronta — se vuoi, un piccolo schema concettuale SOAP (atomo → ambiente
locale → vettore), facoltativo.
SPEAKER NOTES:
"Partiamo dal caso più semplice: sistemi con un solo tipo di atomo. Per ogni struttura ci
serve un descrittore globale — un vettore di numeri che la rappresenti. Il mio punto di
partenza sono stati proprio i descrittori che avevano già funzionato: i SOAP, calcolati atomo
per atomo e poi mediati per ottenere un descrittore dell'intera struttura. Questo ci dà
automaticamente le invarianze che ci servono: alla permutazione degli atomi — non importa in
che ordine li elenco — alla rotazione e traslazione della struttura, e alla scelta della cella
periodica con cui la rappresento.
Una parte che ho considerato utile del progetto è la possibilità di esplorare visivamente
questo spazio a bassa dimensione: quando una struttura è in una zona di transizione, o la sua
fase non è ben definita, è utile poter vedere 'dove si trova' e verso quale simmetria tende —
non solo ottenere un'etichetta secca."

---
SLIDE 7 — Cosa ci serve: encoder + classifier
Bullet:
- Encoder → rappresentazione compatta della struttura
- Classifier → etichetta di simmetria a partire da quella rappresentazione
- Perché non Autoencoder/VAE: decoder e capacità di ricostruzione sono superflui, interessa
  solo separare simmetrie diverse tra loro, non ricostruire l'input
Nessuna figura — slide di testo/transizione.
SPEAKER NOTES:
"Quindi le due cose che ci servono sono: un encoder, che riduce la struttura a questa
rappresentazione compatta, e un classificatore, che le assegna un'etichetta di simmetria.
Il primo pensiero, parlando di encoder, va di solito ad autoencoder o alla sua variante
variazionale. Ma nel nostro caso la capacità di ricostruzione — quindi tutto il decoder — è
completamente superflua. Quello che ci interessa davvero è separare simmetrie diverse tra
loro, non ricostruire l'input."

---
SLIDE 8 — Come funziona la loss SupCon
Questa è la slide più tecnica e più densa: la formula è il contenuto visivo principale.
Layout: formula al centro, ben tipografata (usa l'equation editor), legenda piccola sotto,
eventuale mini-schema dell'esempio a lato o sotto.
FORMULA (rendila con notazione matematica vera, apici/pedici reali):
Per ogni ancora i nel batch, con P(i) = insieme dei positivi (altri elementi del batch con la
stessa etichetta), A(i) = tutti gli altri elementi del batch:

L = media su i di [ -(1/|P(i)|) · Σ (per p in P(i)) di log( exp(sim(z_i, z_p)/τ) / Σ (per a in
A(i)) di exp(sim(z_i, z_a)/τ) ) ]

con sim(z_i, z_j) = (z_i / ||z_i||) · (z_j / ||z_j||)  [similarità coseno tra vettori
normalizzati]

Legenda sotto la formula: "positivo = stessa simmetria nel batch · negativo = simmetria
diversa nel batch · τ = temperatura"
DIAGRAMMA DA CREARE (piccolo, accanto o sotto la formula): un batch schematico con 3 punti di
un colore (cubiche) e 2 punti di un altro colore (esagonali); per una delle cubiche, frecce
verdi "avvicina" verso le altre 2 cubiche (positivi) e frecce rosse "allontana" verso le 2
esagonali (negativi).
SPEAKER NOTES:
"L'architettura che abbiamo scelto è quella del Supervised Contrastive Learning, SupCon.
Voglio spendere un minuto in più sulla teoria, perché è il cuore di tutto il progetto.
L'idea, rispetto a un classificatore normale, è diversa: invece di imparare direttamente a
predire un'etichetta, il modello impara a costruire uno spazio in cui strutture della stessa
simmetria stanno vicine tra loro, e strutture di simmetrie diverse stanno lontane. La loss che
fa questo è definita così: per ogni struttura i del batch — la chiamiamo 'ancora' — guardiamo
gli altri elementi dello stesso batch che condividono la sua etichetta, li chiamiamo i suoi
'positivi', e tutti gli altri, che chiamiamo i suoi 'negativi'. La loss è, mediata su tutte le
ancore del batch: meno il logaritmo del rapporto tra la somma delle similarità (esponenziate e
scalate per una temperatura tau) dell'ancora con i suoi positivi, e la stessa somma estesa a
tutti — positivi e negativi insieme.
In pratica: la loss spinge la similarità tra l'ancora e i suoi positivi verso l'alto, e quella
con i negativi verso il basso, tutto insieme in un solo softmax.
Facciamo un esempio concreto. Immaginate un batch con tre strutture cubiche e due esagonali.
Per una delle strutture cubiche, i suoi positivi sono le altre due cubiche nel batch; i suoi
negativi sono le due esagonali. La loss spinge il suo embedding ad avvicinarsi ai due positivi
e ad allontanarsi dai due negativi — e la stessa cosa succede, contemporaneamente, per ogni
altra struttura del batch, cubica o esagonale che sia.
C'è anche un parametro di temperatura, tau, che controlla quanto la loss sia 'severa': con tau
basso, anche piccole differenze di similarità pesano moltissimo nel softmax, quindi il modello
si concentra soprattutto sui casi più difficili — i vicini scomodi; con tau più alto, il
gradiente è più uniforme su tutti i punti. Nel nostro caso misuriamo la similarità col coseno
tra i vettori — normalizzati, quindi conta solo la direzione, non la lunghezza — perché in
pratica dà risultati più stabili della distanza euclidea diretta."

---
SLIDE 9 — Architettura: SupCon a 4 blocchi, in due fasi
DIAGRAMMA DA CREARE (contenuto principale della slide, box e frecce native di PowerPoint):
un box centrale "Encoder (MLP)" con etichetta piccola "Fase 1"; una freccia verso un box
"Projection Tail (MLP)" (anche questo etichettato "Fase 1", con una nota "loss SupCon
calcolata qui"); poi, separatamente, due frecce dall'encoder verso due box "Classification
Tail (MLP)" e "Visualization Tail (MLP)", entrambi etichettati "Fase 2" con una nota "corpo
congelato". Usa un colore/sfondo diverso per raggruppare visivamente Fase 1 vs Fase 2.
Bullet minimi sotto o accanto al diagramma:
- Fase 1: encoder + projection tail allenati insieme, con la loss SupCon della slide
  precedente
- Fase 2: corpo congelato, classification tail (softmax) e visualization tail (di nuovo loss
  SupCon, proiettata a 2-3D) allenate separatamente
- Nota implementativa: tutti e 4 i blocchi sono semplici MLP (Dense + ReLU) sul vettore SOAP a
  lunghezza fissa — nessuna convoluzione, nessuna rete a grafo. Scelta di partenza semplice,
  non definitiva.
SPEAKER NOTES:
"Prendendo spunto proprio da questo articolo, abbiamo pensato il nostro modello come quattro
blocchi che si possono montare insieme, usati durante un training diviso in due fasi separate.
Nella prima fase alleniamo insieme encoder e projection tail, usando proprio la loss SupCon
che abbiamo appena visto. L'encoder è il corpo principale di parametri, e il suo output viene
poi usato da tutti gli altri blocchi. La projection tail riduce ulteriormente questo output,
ed è sul suo spazio — non direttamente su quello dell'encoder — che viene calcolata la loss: è
come un'estensione dell'encoder, usata solo durante il training e poi scartata.
Nella seconda fase, col corpo ormai congelato, alleniamo separatamente altre due tail: una
classification tail, con una normale softmax, e una visualization tail, che di nuovo usa la
loss SupCon — stessa idea, ma proiettando questa volta a due o tre dimensioni, così da poter
plottare direttamente i cluster di simmetria.
Dal punto di vista implementativo, per essere chiari, tutti e quattro questi blocchi sono
semplicissimi: sono MLP, pochi strati completamente connessi con una ReLU come non linearità —
niente di convoluzionale, niente reti a grafo. Lavorano tutti direttamente sul vettore SOAP a
lunghezza fissa di cui parlavamo prima, non sulla struttura atomica grezza. Ed è voluto:
all'inizio abbiamo scelto la cosa più semplice possibile, solo per capire se l'idea regge,
prima di complicarci la vita con un'architettura più sofisticata. Il passo successivo, che
vedremo tra poco, è proprio sostituire questi MLP con un embedding già pre-addestrato."

---
SLIDE 10 — Un proof-of-concept, non un sistema finito
Bullet:
- Dataset generato sinteticamente con pyxtal (libreria Python), 7 famiglie cristalline, con
  rumore aggiunto artificialmente
- Primo tentativo (spacegroup diretto, in un solo passo) non ha funzionato: troppa varietà
- Soluzione: due step — classificatore di famiglia, poi esperto di spacegroup per famiglia
- Entrambi gli step riusano gli stessi 4 blocchi della slide precedente (cambia solo il
  numero/larghezza degli hidden layer di ciascun MLP)
Nessuna figura — slide di solo testo, tono esplicitamente onesto ("lavoro in corso").
SPEAKER NOTES:
"Passiamo a cosa abbiamo fatto in pratica — e qui vorrei essere onesto: quello che segue non è
un sistema finito, è un proof-of-concept che mostra che l'idea regge, con alcuni limiti che
stiamo ancora capendo.
Il dataset lo generiamo sinteticamente con una libreria python: pyxtal. Questo ci permette di
generare migliaia di strutture per ciascuna delle sette famiglie cristalline, alle quali
aggiungiamo artificialmente rumore — piccoli spostamenti degli atomi, vacanze — per abituare
il modello proprio al tipo di rumore che troviamo nelle traiettorie reali.
Il primo tentativo è stato provare a indovinare direttamente lo spacegroup in un solo
passaggio. Non ha funzionato bene: c'è troppa varietà, decine di spacegroup possibili. Quindi
abbiamo adottato un processo a due step: prima un classificatore di famiglia cristallina,
sette classi; poi, per ogni famiglia, un modello 'esperto' dedicato solo allo spacegroup di
quella famiglia. I due passaggi partono dallo stesso descrittore SOAP e sono completamente
indipendenti tra loro. Entrambi, però, sono costruiti con gli stessi quattro blocchi visti
prima — cambia solo quanti strati nascosti diamo a ciascun MLP e quanto sono larghi."

---
SLIDE 11 — Il classificatore di famiglia: separazione netta
Bullet:
- Mappa 2D (blocco visualization) → separazione per famiglia evidente
- Confusion matrix del classificatore → forte diagonalità, accuratezza media ~97%
FIGURA ALLEGATA 1: mappa 2D famiglia (immagine allegata — se manca, "[FIGURA MANCANTE: mappa
2D famiglia]")
FIGURA ALLEGATA 2: confusion matrix famiglia (immagine allegata — se manca, "[FIGURA MANCANTE:
confusion matrix famiglia]")
Disponi le due figure affiancate, stessa altezza, con la mappa a sinistra e la confusion
matrix a destra.
SPEAKER NOTES:
"Passiamo a qualche risultato concreto di training. Il dataset è quello di cui vi ho appena
parlato: le strutture sintetiche generate con pyxtal sulle sette famiglie, con il rumore
artificiale aggiunto. Su questo dataset abbiamo allenato sia il classificatore di famiglia,
sia, per ciascuna famiglia, l'esperto di spacegroup.
Qui vedete una mappa a due dimensioni delle nostre strutture, il risultato del blocco di
visualizzazione: ogni punto è una struttura separata, colorata secondo la sua famiglia. Come
vedete, la separazione nello spazio per famiglia è evidente. Riporto anche il risultato del
classificatore vero e proprio, la confusion matrix: mostra una forte diagonalità, con
un'accuratezza media di circa il 97%.
A questo punto, le strutture identificate con una certa famiglia vengono passate all'esperto
corrispondente, per classificarne lo spacegroup."
NOTA PER TE (Claude Design): il numero "97%" è un'approssimazione indicativa — se hai
un'immagine della confusion matrix che riporta un numero esatto, usa quello sia nel testo
della slide sia nelle speaker notes al posto di "circa il 97%".

---
SLIDE 12 — Per famiglia, uno spacegroup expert — con un'eccezione
Bullet:
- Ogni famiglia ha il proprio esperto di spacegroup (stesso descrittore SOAP, esperti
  completamente indipendenti tra loro)
- Separazione ancora evidente per quasi tutte le famiglie — tranne Cubic
- Stessa situazione nelle confusion matrix: l'unica a dare problemi è Cubic
- Sospetto (non confermato): l'input, il descrittore SOAP stesso per quella famiglia — non il
  modello. Indagine ancora aperta.
FIGURA ALLEGATA 1: griglia con le mappe 2D spacegroup delle 7 famiglie insieme (immagine
allegata — se manca, "[FIGURA MANCANTE: griglia mappe spacegroup 7 famiglie]")
FIGURA ALLEGATA 2 e 3: due confusion matrix per esperto, affiancate come contrasto — una è
quella della famiglia Cubic (problematica), l'altra è di una famiglia qualunque tra le altre 6
(pulita, forte diagonale) — usa le immagini allegate; se ne manca anche solo una, riquadro
"[FIGURA MANCANTE: confusion matrix esperto <nome famiglia>]".
Se lo spazio in slide non basta per tutte e tre le figure con leggibilità decente, tieni la
griglia spacegroup (figura 1) grande e le due confusion matrix di contrasto più piccole sotto
o a lato — priorità alla leggibilità, non a stare tutti sulla stessa riga.
SPEAKER NOTES:
"Qui vedete il plot generato dalla visualization tail per ciascuna famiglia. Come vedete, la
separazione in gruppi resta evidente per quasi tutte — con un'eccezione: il cubic.
Stessa situazione la troviamo nelle confusion matrix: ottima diagonalità per ogni famiglia
tranne il cubic.
Dai numerosi test fatti finora, il problema non sembra dipendere dal modello, ma più
probabilmente dall'input — dal descrittore SOAP stesso, per quella famiglia specifica. Su
questo sto ancora indagando, quindi non ho ancora una risposta definitiva da darvi oggi."

---
SLIDE 13 — Prossimo passo: embedding MACE
Bullet:
- Sostituire SOAP + encoder allenato con l'embedding di un modello MACE già pre-addestrato
- Cattura nativamente interazioni angolari/a molti corpi, non solo l'ambiente locale attorno
  a un singolo atomo
- Ipotesi da testare, anche sui casi più difficili incontrati finora — non una soluzione già
  in mano
- Nessun run MACE ancora eseguito: prossimo step concreto, non ancora fatto
Nessuna figura reale — se vuoi, un semplice schema "SOAP + encoder (oggi)" → freccia →
"embedding MACE pre-addestrato, congelato (prossimo passo)".
SPEAKER NOTES:
"Come prossimo passo, l'idea è sostituire la coppia 'SOAP più encoder allenato' con
l'embedding di un modello MACE già pre-addestrato che, come sappiamo, cattura in modo nativo
anche le interazioni angolari, non solo l'ambiente locale attorno a un singolo atomo. È
un'ipotesi che vogliamo provare, anche per vedere se aiuta sui casi più difficili che abbiamo
incontrato finora — non vi prometto che li risolverà, ma è la direzione in cui stiamo
andando."

---
SLIDE 14 — Conclusioni
Bullet (recap in 4 punti, questo è il contenuto principale della slide):
- Problema: strutture da Nested Sampling troppo rumorose per gli strumenti classici
- Approccio: SupCon su descrittori SOAP, classificazione famiglia → esperto di spacegroup, da
  strutture generate al computer
- Funziona per la maggior parte delle famiglie; resta un'eccezione aperta (Cubic)
- Prossimo passo: embedding MACE
Un "Grazie" piccolo in fondo alla slide (non come titolo principale).
Nessuna figura.
SPEAKER NOTES:
"Per riassumere. Siamo partiti da un problema concreto: le strutture del Nested Sampling sono
troppo rumorose per gli strumenti classici di identificazione della simmetria. Abbiamo provato
ad affrontarlo allenando un modello — SupCon su descrittori SOAP — a riconoscere famiglia e
spacegroup direttamente da strutture generate al computer, senza bisogno di riferimenti noti.
Funziona, in buona parte: la separazione per famiglia è netta, e la maggior parte degli
esperti di spacegroup classifica bene. Resta un'eccezione aperta, quella cubica, su cui stiamo
ancora lavorando. E il prossimo passo naturale è provare un embedding più ricco, con MACE.
È esattamente questo lo stato del progetto: un'idea che regge, con dei limiti chiari e ben
identificati. Grazie, sono felice di rispondere a domande."

---

Alla fine, genera il file .pptx scaricabile e riepiloga in una lista quali slide contengono
segnaposto "[FIGURA MANCANTE: ...]" da sostituire a mano, così so subito cosa completare
prima di presentare.
```

---

## 3. Dopo la generazione

Una volta ottenuto il `.pptx`, controlla in particolare:
- Che le speaker notes siano finite davvero nel campo note di PowerPoint (non ripetute in
  slide) — è il punto più facile da sbagliare per un tool di generazione automatica.
- Che la formula in slide 8 sia leggibile e non sia diventata un'immagine sgranata o testo
  Unicode illeggibile.
- Il numero di accuratezza in slide 11 (~97%, vedi nota nel prompt) — conferma quello esatto
  sulla confusion matrix che hai effettivamente allegato, prima di tenerlo o correggerlo.
- I segnaposto "[FIGURA MANCANTE: ...]", se ce ne sono — vanno sostituiti con le immagini vere
  prima di presentare.
