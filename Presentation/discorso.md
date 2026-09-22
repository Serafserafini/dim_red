# Discorso — bozza v1 (~20 minuti)

Testo pensato per essere **letto ad alta voce**, non per essere proiettato: sulle slide va
il brief sintetico in `brief_slides_claude_design.md`, qui va quello che dici tu. Registro
colloquiale, prima persona plurale ("abbiamo", "ci serve"), come nel tuo brainstorm
originale. `[SLIDE: ...]` indica dove cambiare slide — sono punti di riferimento, non
vincoli rigidi.

**Questa è una bozza per iniziare a lavorarci insieme** — dimmi dove non ti riconosci nel
tono, dove vuoi tagliare, dove vuoi aggiungere un dettaglio personale (soprattutto nelle
sezioni 1 e 3, che vengono da lavoro tuo che io non ho potuto leggere da nessuna parte) e
lo sistemiamo insieme frase per frase.

Tempo stimato: ~2600 parole, ~130-140 parole/minuto con pause per slide → **circa 19-20
minuti**.

---

## [SLIDE 1: Titolo] — 0:00

Buongiorno a tutti. Oggi vi parlo di un'idea su cui stiamo lavorando da qualche mese: usare
il machine learning per riconoscere automaticamente la simmetria cristallina di una
struttura, in situazioni dove i metodi classici non ce la fanno. Vi anticipo subito che non
è un lavoro finito — è più un'idea che sta prendendo forma, e oggi voglio condividere con
voi a che punto siamo, compresi i punti che non abbiamo ancora chiariti del tutto.

## [SLIDE 2: Motivazione — nested sampling / brass] — 0:30

Il problema da cui siamo partiti nasce dal Nested Sampling. Come sapete, il Nested Sampling
ci dà una traiettoria fatta di moltissime strutture, ciascuna rappresentativa del sistema a
una certa temperatura. L'idea è: possiamo usare queste strutture per costruire il diagramma
di fase — cioè per dire "in questa regione di temperatura e composizione, il sistema si
trova in questa fase, con questa simmetria". Un esempio concreto che useremo per tutto il
discorso è l'ottone, il brass, che ha diverse fasi — alfa, beta, gamma — ciascuna con una
propria simmetria cristallina.

## [SLIDE 3: Il problema — perché serve altro] — ~2:00

Il problema è che queste strutture non sono pulite. Vengono da mosse Monte Carlo a
temperatura finita, quindi sono rumorose: gli atomi sono spostati rispetto ai punti esatti
di simmetria, e ci sono vacanze atomiche. Questo significa che le librerie classiche per
identificare la simmetria cristallina — penso a spglib — semplicemente non funzionano su
input così rumorosi. E anche provando a "ripulire" la struttura con un'ottimizzazione
energetica locale, il problema resta: l'ottimizzazione si ferma al minimo locale più
vicino, che spesso non è nemmeno la fase giusta. Lo stesso discorso vale per simulazioni di
dinamica molecolare ad alta temperatura.

Quindi ci serve qualcos'altro.

## [SLIDE 4: Lavoro precedente — RDF/Steinhardt + UMAP] — ~3:30

Il primo tentativo che abbiamo fatto è stato più diretto: analizzare le strutture con
parametri strutturali classici, come la RDF o i parametri di Steinhardt. In alternativa,
abbiamo preso le strutture più rappresentative per una certa coppia di temperatura e
composizione, le abbiamo rilassate con una breve simulazione di annealing, e poi le abbiamo
confrontate tra loro proiettandole in una mappa bidimensionale con UMAP.

Questo ha funzionato sorprendentemente bene: nella mappa si formavano dei cluster netti,
che corrispondevano proprio a regioni precise di temperatura e composizione. E quando
abbiamo proiettato nello stesso spazio le strutture di riferimento delle fasi alfa, beta e
gamma del brass, prese dalla letteratura, sono cadute esattamente dentro quei cluster.

## [SLIDE 5: La domanda che porta a dim_red] — ~5:30

Questo però ci ha lasciato con una domanda: e se non avessimo strutture di riferimento con
cui confrontare? Il brass lo conosciamo bene, ma per un sistema nuovo non abbiamo nessuna
fase alfa-beta-gamma già pronta in letteratura da proiettare.

Quello che ci serve, allora, è qualcosa che porti già con sé una conoscenza delle simmetrie
cristalline in generale — indipendente dal nostro sistema chimico specifico — e che sia
robusto al rumore termico e alle vacanze atomiche. L'idea è allenare un modello di machine
learning a riconoscere lo spacegroup, usando strutture generabili al computer, non
raccolte a mano.

## [SLIDE 6: Intro tecnica — SOAP e visualizzazione] — ~7:00

Partiamo dal caso più semplice: sistemi con un solo tipo di atomo. Per ogni struttura ci
serve un descrittore globale — un vettore di numeri che la rappresenti. La strategia più
semplice è usare i descrittori SOAP calcolati atomo per atomo, e poi mediarli per ottenere
un descrittore dell'intera struttura. Questo ci dà automaticamente invarianza per
permutazione degli atomi, per rotazione, e così via.

Una parte importante del progetto è poter esplorare *visivamente* questo spazio a bassa
dimensione: quando una struttura è in una zona di transizione, o la sua fase non è ben
definita, è utile poter vedere "dove si trova" e verso quale simmetria tende — non solo
ottenere un'etichetta secca.

## [SLIDE 7: Cosa serve — encoder + classifier, perché non AE/VAE] — ~8:30

Quindi le due cose che ci servono sono: un **encoder**, che riduce la struttura a questa
rappresentazione compatta, e un **classificatore**, che le assegna un'etichetta di
simmetria.

Il primo pensiero, parlando di encoder, va di solito ad autoencoder o alla sua variante
variazionale. Ma nel nostro caso la capacità di ricostruzione — quindi tutto il decoder — è
completamente superflua. Quello che ci interessa davvero è separare simmetrie diverse tra
loro, non ricostruire l'input.

## [SLIDE 8: Architettura SupCon — 4 blocchi] — ~10:00

L'architettura che abbiamo scelto è quella del Supervised Contrastive Learning, SupCon — la
sua loss ha esattamente l'obiettivo di separare le classi in uno spazio a bassa dimensione.
Prendendo spunto dall'articolo originale, abbiamo pensato il nostro modello come quattro
blocchi che si possono montare insieme.

Il primo è l'**encoder**: è il corpo principale di parametri, allenato nella fase iniziale,
e il suo output viene usato da tutti gli altri blocchi.

Il secondo è la **projection tail**: riduce ulteriormente l'output dell'encoder, e viene
allenata insieme a lui nella fase iniziale — è proprio su questo spazio che viene calcolata
la loss SupCon.

Poi, in una seconda fase, con il corpo ormai congelato, attacchiamo altri due blocchi: una
**classification tail**, che prende l'output dell'encoder e lo classifica con una normale
softmax, e una **visualization tail**, che invece riduce l'output a due o tre dimensioni
per poterlo plottare e vedere i cluster di simmetria.

## [SLIDE 9: Risultati — framing onesto + dataset + due step] — ~13:00

Passiamo a cosa abbiamo fatto in pratica — e qui vorrei essere onesto: quello che segue non
è un sistema finito, è un proof-of-concept che mostra che l'idea regge, con alcuni limiti
che stiamo ancora capendo.

Il dataset lo generiamo sinteticamente con uno strumento chiamato pyxtal, invece che
raccoglierlo — coerente con quello che dicevamo prima: ci serve una conoscenza indipendente
dal sistema chimico specifico. Generiamo migliaia di strutture per ciascuna delle sette
famiglie cristalline, e aggiungiamo artificialmente rumore — piccoli spostamenti degli
atomi, vacanze — per abituare il modello proprio al tipo di rumore che troviamo nelle
traiettorie reali.

Il primo tentativo è stato provare a indovinare direttamente lo spacegroup in un solo
passaggio. Non ha funzionato bene: c'è troppa varietà, decine di spacegroup possibili.
Quindi abbiamo adottato un processo a due step: prima un classificatore di famiglia
cristallina, sette classi; poi, per ogni famiglia, un modello "esperto" dedicato solo allo
spacegroup di quella famiglia. I due passaggi partono dallo stesso descrittore SOAP e sono
completamente indipendenti tra loro.

## [SLIDE 10: Risultati — la mappa 2D] — ~15:30

Qui vedete una mappa a due dimensioni delle nostre strutture: ogni punto è una struttura, e
i colori corrispondono alle diverse simmetrie. Il passo di famiglia funziona bene, e anche
il passo di spacegroup funziona bene per la maggior parte delle famiglie.

## [SLIDE 11: Risultati — l'eccezione Cubic] — ~16:30

C'è però un'eccezione: una famiglia, quella cubica, ci dà sistematicamente più filo da
torcere delle altre. Non sembra dipendere dal modello — cambiando architettura o capacità
il problema resta identico — il sospetto è che c'entri più l'input, o il training set, per
quella famiglia specifica, che il modello in sé. È qualcosa che stiamo ancora indagando, e
non ho ancora una spiegazione di cui mi senta sicuro abbastanza da darvela oggi.

## [SLIDE 12: Futuro — MACE] — ~18:00

Come prossimo passo, l'idea è sostituire la coppia "SOAP più encoder allenato" con
l'embedding di un modello **MACE** già pre-addestrato — un modello equivariante che cattura
in modo nativo anche le interazioni angolari, non solo l'ambiente locale attorno a un
singolo atomo. È un'ipotesi che vogliamo provare, anche per vedere se aiuta sui casi più
difficili che abbiamo incontrato finora — non vi prometto che li risolverà, ma è la
direzione in cui stiamo andando.

## [SLIDE 13: Chiusura] — ~19:00

In generale, questo è esattamente lo stato del progetto: un'idea che funziona in linea di
principio, con dei limiti chiari che stiamo ancora esplorando. Grazie, e sono felice di
rispondere a domande.

---

## Note per la revisione insieme

- **Sezioni 2 e 4** (motivazione nested sampling/brass, lavoro precedente RDF/UMAP): scritte
  solo sulla base di quanto già nel brainstorm — se hai dettagli/numeri/aneddoti in più
  (quanti cluster, quali T/composizioni, come si chiamava il sistema oltre al brass) è qui
  che vanno aggiunti, io non ho materiale ulteriore da cui attingere.
- **Slide 10** (mappa 2D): il testo presume che tu mostri una figura concreta — vedi
  `brief_slides_claude_design.md` per quale.
- **Slide 11** (Cubic): rispetta il vincolo che avevi dato — un accenno, nessuna causa-radice.
  Se in fase di prova ti senti a disagio anche solo con "il sospetto è che c'entri più
  l'input... che il modello in sé", possiamo accorciarla ulteriormente a una frase sola
  ("una famiglia resta sistematicamente più difficile delle altre, ci stiamo ancora
  lavorando") — dimmelo.
- **Domande da anticipare** (non nel discorso, ma utile prepararle a mente): "perché non
  avete provato spglib con tolleranza aumentata?", "quanto costa/quanto è lontano un run
  MACE reale?", "quanto è realistico pyxtal come proxy di strutture rumorose vere?".
