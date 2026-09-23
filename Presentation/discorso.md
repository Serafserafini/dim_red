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

Tempo stimato: ~3300 parole (aggiornato dopo la slide 8 sulla teoria SupCon e
l'espansione di 11-12 su classificatore famiglia/esperti spacegroup),
~130-140 parole/minuto con pause per slide → **circa 23-25 minuti**, sopra i 20 di target.
Candidati per tagliare, in ordine di impatto: l'esempio delle 3 cubiche/2 esagonali in
slide 8 (~1 min), la frase su tau in slide 8 (~30s), accorciare slide 12 a una sola
menzione di Cubic invece di viz+confusion matrix separate (~30s). Vedi anche la nota
originale sulla slide 8 più sotto.

---

## [SLIDE 1: Titolo] — 0:00

Buongiorno a tutti. Oggi vi parlo di un'idea su cui stiamo lavorando da qualche mese: usare
il machine learning per riconoscere automaticamente la simmetria cristallina di una
struttura, in situazioni dove i metodi classici non ce la fanno. Vi anticipo subito che non
è un lavoro finito — è più un'idea che sta prendendo forma, e oggi voglio condividere con
voi a che punto siamo, compresi i punti che non abbiamo ancora chiariti del tutto.

## [SLIDE 2: Motivazione — nested sampling / brass] — 0:30

Il problema da cui siamo partiti nasce dal Nested Sampling — lo conoscete già, quindi solo
un richiamo veloce: è un metodo di campionamento che esplora lo spazio delle configurazioni
di un sistema, e come sottoprodotto ci dà accesso diretto alla funzione di partizione su un
ampio intervallo di temperature. Il punto per noi è che ci restituisce una traiettoria
fatta di moltissime strutture, ciascuna rappresentativa del sistema a una certa
temperatura. L'idea è: possiamo usare queste strutture per costruire il diagramma di fase —
cioè per dire "in questa regione di temperatura e composizione, il sistema si trova in
questa fase, con questa simmetria". Un esempio concreto, che ci servirà tra poco per
spiegare cosa avevamo già provato e perché non bastava, è l'ottone, il brass, che ha
diverse fasi — alfa, beta, gamma — ciascuna con una propria simmetria cristallina.

## [SLIDE 3: Il problema — perché serve altro] — ~2:00

Il problema è che queste strutture non sono pulite. Vengono da mosse Monte Carlo a
temperatura finita, quindi sono rumorose: gli atomi sono spostati rispetto ai punti esatti
di simmetria, e potrebbero esserci vacanze atomiche. Nelle traiettorie del Nested Sampling,
poi, compaiono anche strutture che non sono nemmeno vicine a un minimo locale sulla
superficie di energia potenziale. Questo significa che le librerie classiche per
identificare la simmetria cristallina — penso a spglib — semplicemente non funzionano su
input così rumorosi.

Si potrebbe pensare di "ripulire" ogni struttura con un'ottimizzazione energetica locale
prima di analizzarla, ma qui ci sono due problemi. Il primo è di costo: non possiamo
permetterci di ottimizzare tutte le strutture della traiettoria, quindi al massimo possiamo
farlo solo su una parte di esse. Il secondo, più di fondo, è che anche quando lo facciamo,
l'ottimizzazione si ferma al minimo locale più vicino sulla superficie di energia
potenziale — e non c'è nessuna garanzia che quel minimo corrisponda alla fase reale del
sistema.

Quindi ci serve qualcos'altro.

## [SLIDE 4: Lavoro precedente — SOAP + UMAP] — ~3:30

L'unico approccio che ci ha dato risultati concreti è stato questo: abbiamo preso le
strutture più rappresentative per una certa coppia di temperatura e composizione, le
abbiamo sottoposte a una simulazione di dinamica molecolare annealed, e, usando UMAP per
confrontare i descrittori SOAP delle strutture viste durante questa simulazione, abbiamo
costruito una mappa 2D.

Questo ha funzionato sorprendentemente bene: nella mappa si formavano dei cluster netti,
che corrispondevano proprio a regioni precise di temperatura e composizione. E quando
abbiamo proiettato nello stesso spazio le strutture di riferimento delle fasi alfa, beta e
gamma del brass, prese dalla letteratura, sono cadute esattamente dentro quei cluster.

[FIGURA: il grafico dei cluster esiste già — va solo recuperato da dove l'avete salvato]

## [SLIDE 5: La domanda che porta a dim_red] — ~5:30

Questo però ci ha lasciato con una domanda: e se non avessimo strutture di riferimento con
cui confrontare? Il brass lo conosciamo bene, ma per un sistema nuovo potremmo non avere nessuna
fase alfa-beta-gamma già pronta in letteratura da proiettare.

Quello che ci serve, allora, è qualcosa che porti già con sé una conoscenza delle simmetrie
cristalline in generale — indipendente dal nostro sistema chimico specifico — e che sia
robusto al rumore termico e alle vacanze atomiche. L'idea è allenare un modello di machine
learning a riconoscere lo spacegroup, usando strutture la cui simmetria é nota e sicura.

## [SLIDE 6: Intro tecnica — SOAP e visualizzazione] — ~7:00

Partiamo dal caso più semplice: sistemi con un solo tipo di atomo. Per ogni struttura ci
serve un descrittore globale — un vettore di numeri che la rappresenti. Il mio punto di
partenza sono stati proprio i descrittori che avevano già funzionato: i SOAP, calcolati
atomo per atomo e poi mediati per ottenere un descrittore dell'intera struttura. Questo ci
dà automaticamente le invarianze che ci servono: alla permutazione degli atomi — non
importa in che ordine li elenco — alla rotazione e traslazione della struttura, e alla
scelta della cella periodica con cui la rappresento.

Una parte che ho considerato utile del progetto è la possibilità di esplorare *visivamente*
questo spazio a bassa dimensione: quando una struttura è in una zona di transizione, o la
sua fase non è ben definita, è utile poter vedere "dove si trova" e verso quale simmetria
tende — non solo ottenere un'etichetta secca.

## [SLIDE 7: Cosa serve — encoder + classifier, perché non AE/VAE] — ~8:30

Quindi le due cose che ci servono sono: un **encoder**, che riduce la struttura a questa
rappresentazione compatta, e un **classificatore**, che le assegna un'etichetta di
simmetria.

Il primo pensiero, parlando di encoder, va di solito ad autoencoder o alla sua variante
variazionale. Ma nel nostro caso la capacità di ricostruzione — quindi tutto il decoder — è
completamente superflua. Quello che ci interessa davvero è separare simmetrie diverse tra
loro, non ricostruire l'input.

## [SLIDE 8: SupCon — la teoria] — ~10:00

L'architettura che abbiamo scelto è quella del Supervised Contrastive Learning, SupCon.
Voglio spendere un minuto in più sulla teoria, perché è il cuore di tutto il progetto.

L'idea, rispetto a un classificatore normale, è diversa: invece di imparare direttamente a
predire un'etichetta, il modello impara a costruire uno spazio in cui strutture della
stessa simmetria stanno vicine tra loro, e strutture di simmetrie diverse stanno lontane.
[SLIDE: formula] La loss che fa questo è definita così:

> per ogni struttura *i* del batch — la chiamiamo "ancora" — guardiamo gli altri elementi
> dello stesso batch che condividono la sua etichetta, li chiamiamo i suoi "positivi", e
> tutti gli altri, che chiamiamo i suoi "negativi". La loss è, mediata su tutte le ancore
> del batch: meno il logaritmo del rapporto tra la somma delle similarità (esponenziate e
> scalate per una temperatura tau) dell'ancora con i suoi positivi, e la stessa somma
> estesa a tutti — positivi e negativi insieme.

In pratica: la loss spinge la similarità tra l'ancora e i suoi positivi verso l'alto, e
quella con i negativi verso il basso, tutto insieme in un solo softmax.

Facciamo un esempio concreto. Immaginate un batch con tre strutture cubiche e due
esagonali. Per una delle strutture cubiche, i suoi positivi sono le altre due cubiche nel
batch; i suoi negativi sono le due esagonali. La loss spinge il suo embedding ad
avvicinarsi ai due positivi e ad allontanarsi dai due negativi — e la stessa cosa succede,
contemporaneamente, per ogni altra struttura del batch, cubica o esagonale che sia.

C'è anche un parametro di temperatura, tau, che controlla quanto la loss sia "severa": con
tau basso, anche piccole differenze di similarità pesano moltissimo nel softmax, quindi il
modello si concentra soprattutto sui casi più difficili — i vicini scomodi; con tau più
alto, il gradiente è più uniforme su tutti i punti. Nel nostro caso misuriamo la similarità
col coseno tra i vettori — normalizzati, quindi conta solo la direzione, non la lunghezza —
perché in pratica dà risultati più stabili della distanza euclidea diretta.

## [SLIDE 9: Architettura SupCon — 4 blocchi] — ~12:00

Prendendo spunto proprio da questo articolo, abbiamo pensato il nostro modello come quattro
blocchi che si possono montare insieme, usati durante un training diviso in due fasi separate.

Nella prima fase alleniamo insieme **encoder** e **projection tail**, usando proprio la
loss SupCon che abbiamo appena visto. L'encoder è il corpo principale di parametri, e il
suo output viene poi usato da tutti gli altri blocchi. La projection tail riduce
ulteriormente questo output, ed è sul suo spazio — non direttamente su quello
dell'encoder — che viene calcolata la loss: è come un'estensione dell'encoder, usata solo
durante il training e poi scartata.

Nella seconda fase, col corpo ormai congelato, alleniamo separatamente altre due tail: una
**classification tail**, con una normale softmax, e una **visualization tail**, che di
nuovo usa la loss SupCon — stessa idea, ma proiettando questa volta a due o tre dimensioni,
così da poter plottare direttamente i cluster di simmetria.

Dal punto di vista implementativo, per essere chiari, tutti e quattro questi blocchi sono
semplicissimi: sono MLP, pochi strati completamente connessi con una ReLU come non
linearità — niente di convoluzionale, niente reti a grafo. Lavorano tutti direttamente sul
vettore SOAP a lunghezza fissa di cui parlavamo prima, non sulla struttura atomica grezza.
Ed è voluto: all'inizio abbiamo scelto la cosa più semplice possibile, solo per capire se
l'idea regge, prima di complicarci la vita con un'architettura più sofisticata. Il passo
successivo, che vedremo tra poco, è proprio sostituire questi MLP con un embedding già
pre-addestrato.

## [SLIDE 10: Risultati — framing onesto + dataset + due step] — ~13:30

Passiamo a cosa abbiamo fatto in pratica — e qui vorrei essere onesto: quello che segue non
è un sistema finito, è un proof-of-concept che mostra che l'idea regge, con alcuni limiti
che stiamo ancora capendo.

Il dataset lo generiamo sinteticamente con una libreria python: pyxtal.
Questo ci permette di generare migliaia di strutture per ciascuna delle sette
famiglie cristalline, alle quali aggiungiamo artificialmente rumore — piccoli spostamenti degli
atomi, vacanze — per abituare il modello proprio al tipo di rumore che troviamo nelle
traiettorie reali.

Il primo tentativo è stato provare a indovinare direttamente lo spacegroup in un solo
passaggio. Non ha funzionato bene: c'è troppa varietà, decine di spacegroup possibili.
Quindi abbiamo adottato un processo a due step: prima un classificatore di famiglia
cristallina, sette classi; poi, per ogni famiglia, un modello "esperto" dedicato solo allo
spacegroup di quella famiglia. I due passaggi partono dallo stesso descrittore SOAP e sono
completamente indipendenti tra loro. Entrambi, però, sono costruiti con gli stessi quattro
blocchi visti prima — cambia solo quanti strati nascosti diamo a ciascun MLP e quanto sono
larghi.


## [SLIDE 11: Risultati — il classificatore di famiglia] — ~16:00

Passiamo a qualche risultato concreto di training. Il dataset è quello di cui vi ho appena
parlato: le strutture sintetiche generate con pyxtal sulle sette famiglie, con il rumore
artificiale aggiunto. Su questo dataset abbiamo allenato sia il classificatore di famiglia,
sia, per ciascuna famiglia, l'esperto di spacegroup.

Qui vedete una mappa a due dimensioni delle nostre strutture, il risultato del blocco di
visualizzazione: ogni punto è una struttura separata, colorata secondo la sua famiglia.
Come vedete, la separazione nello spazio per famiglia è evidente. Riporto anche il
risultato del classificatore vero e proprio, la confusion matrix: mostra una forte
diagonalità, con un'accuratezza media di circa il 97%
[NUMERO ESATTO da confermare sulla figura che sceglierai — nella config di riferimento,
`experiments/plots/round7/confusion_matrix_topmodel_val.png`, è 0.9726].

A questo punto, le strutture identificate con una certa famiglia vengono passate
all'esperto corrispondente, per classificarne lo spacegroup.

## [SLIDE 12: Risultati — gli esperti di spacegroup, e l'eccezione Cubic] — ~17:30

Qui vedete il plot generato dalla visualization tail per ciascuna famiglia. Come vedete, la
separazione in gruppi resta evidente per quasi tutte — con un'eccezione: il cubic.

Stessa situazione la troviamo nelle confusion matrix: ottima diagonalità per ogni famiglia
tranne il cubic.

Dai numerosi test fatti finora, il problema non sembra dipendere dal modello, ma più
probabilmente dall'input — dal descrittore SOAP stesso, per quella famiglia specifica. Su
questo sto ancora indagando, quindi non ho ancora una risposta definitiva da darvi oggi.

## [SLIDE 13: Futuro — MACE] — ~20:30

Come prossimo passo, l'idea è sostituire la coppia "SOAP più encoder allenato" con
l'embedding di un modello **MACE** già pre-addestrato che come sappiamo, cattura
in modo nativo anche le interazioni angolari, non solo l'ambiente locale attorno a un
singolo atomo. È un'ipotesi che vogliamo provare, anche per vedere se aiuta sui casi più
difficili che abbiamo incontrato finora — non vi prometto che li risolverà, ma è la
direzione in cui stiamo andando.

## [SLIDE 14: Conclusioni] — ~22:00

Per riassumere. Siamo partiti da un problema concreto: le strutture del Nested Sampling
sono troppo rumorose per gli strumenti classici di identificazione della simmetria.
Abbiamo provato ad affrontarlo allenando un modello — SupCon su descrittori SOAP — a
riconoscere famiglia e spacegroup direttamente da strutture generate al computer, senza
bisogno di riferimenti noti.

Funziona, in buona parte: la separazione per famiglia è netta, e la maggior parte degli
esperti di spacegroup classifica bene. Resta un'eccezione aperta, quella cubica, su cui
stiamo ancora lavorando. E il prossimo passo naturale è provare un embedding più ricco, con
MACE.

È esattamente questo lo stato del progetto: un'idea che regge, con dei limiti chiari e ben
identificati. Grazie, sono felice di rispondere a domande.

---

## Note per la revisione insieme

- **Slide 8** (teoria SupCon): nuova, aggiunta su tua richiesta — formula della loss
  (versione a coseno, quella che usiamo davvero) + esempio concreto. Ha spostato in avanti
  di ~2 minuti tutto quello che segue — il totale ora è ~21-22 minuti anziché 20, vedi nota
  sul tempo qui sopra. Se serve tornare a 20 netti, i candidati più facili da tagliare sono
  l'esempio delle 3 cubiche/2 esagonali (si può raccontare senza, solo con la formula) o la
  frase sulla temperatura tau.
- **Slide 11-12** (riscritte su tuo testo, espanse rispetto alla bozza precedente): ora
  mostrano davvero 4 figure — mappa famiglia, confusion matrix famiglia, mappa
  spacegroup per famiglia, confusion matrix per esperto — vedi `brief_slides_claude_design.md`
  per i path esatti. La menzione di Cubic è leggermente più specifica di prima ("il
  sospetto è ... dal descrittore SOAP stesso" invece del generico "input/training set") —
  resta comunque un sospetto dichiarato, non una causa-radice, quindi coerente col vincolo
  che avevi dato. Il numero di accuratezza in slide 11 è un placeholder da confermare (vedi
  nota inline) — non l'ho inventato, ma nemmeno so con certezza quale run/figura userai.
- Slide 12 ora mostra sia i plot di visualizzazione sia le confusion matrix per ogni
  esperto — se in prova risulta troppo densa per il tempo a disposizione, il taglio più
  naturale è mostrare solo le confusion matrix (più dirette da leggere a colpo d'occhio) e
  lasciare fuori i 7 plot di visualizzazione, o viceversa.
- **Domande da anticipare** (non nel discorso, ma utile prepararle a mente): "perché non
  avete provato spglib con tolleranza aumentata?", "quanto costa/quanto è lontano un run
  MACE reale?", "quanto è realistico pyxtal come proxy di strutture rumorose vere?".
