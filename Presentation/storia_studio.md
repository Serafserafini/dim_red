# La storia dello studio "family_only"

Racconto unico e lineare di tutto quello che è stato fatto in `experiments/` — riorganizzato
per **logica narrativa**, non per ordine cronologico reale dei round (i numeri di round tra
parentesi sono solo riferimenti per andare a riprendere le note complete, non indicano
l'ordine in cui leggere questa pagina). Setup costante in tutto lo studio: strutture
sintetiche generate con pyxtal, 7 famiglie cristalline, classificate soprattutto con
`supcon` (SupCon, contrastive learning supervisionato) su descrittori SOAP. Metrica
primaria: accuratezza di classificazione della famiglia (poi, nella seconda parte della
storia, dello spacegroup). Metrica secondaria: qualità della visualizzazione 2D
(silhouette, k-means ARI/NMI, k-NN accuracy). Regola tenuta ferma ovunque: **mai una
selezione automatica di configurazione "migliore"** — solo numeri, confrontati da un
umano.

---

## Atto 1 — Il punto di partenza: tutti i modelli funzionano, quasi indistinguibili

Il primo confronto a parità di condizioni tra `vae`, `autoencoder` e `supcon` — con
l'identità chimica ancora presente nel dataset (`element_agnostic: false`, 2 specie
chimiche placeholder, SOAP a 12960 dimensioni) — non produce un vincitore: tutti e tre
superano 0.97 di accuratezza famiglia, spesso 0.98-0.99. Il singolo run migliore in
assoluto di tutto lo studio nasce qui: **supcon, distanza cosine, τ=0.1, encoder
`[128,64]`, batching random → 0.9938** *(round 1)*. Mai più superato da nessun esperimento
successivo.

Ma sull'obiettivo secondario — una visualizzazione 2D leggibile — il quadro si ribalta:
con il solo fallback PCA, **vae vince nettamente** (silhouette ~0.32-0.48), mentre supcon è
il peggiore nonostante la sua accuratezza migliore. Il motivo (svelato solo più avanti,
vedi Atto 3) è che supcon, per costruzione, produce un pattern "a stella radiale": i vicini
locali sono puliti (k-NN accuracy ~0.99) ma tutte le famiglie sono ammassate vicino
all'origine su una scala enorme (assi ±1000 contro ±4 di vae) — ottimo per la
classificazione, pessimo per un occhio umano che guarda un grafico.

In questo stesso confronto emergono tre comportamenti patologici, diagnosticati con
precisione:
- **vae con encoder a un solo strato `[64]`** diverge subito a NaN.
- **vae con `beta=2.0` + `lambda_family=0.5`** produce una head di classificazione "morta"
  (loss bloccata a ln(7), cioè predizione uniforme casuale su 7 classi) — confermato
  **sistematico**, non sfortuna di inizializzazione, ripetendo l'esperimento su 3 seed
  diversi.
- **supcon con `distance: euclidean`** collassa (accuratezza 0.18-0.26): la norma della
  proiezione esplode senza controllo quando `lambda_norm=0`. Sarà un tema ricorrente:
  **euclidean perde sempre contro cosine come loss del corpo**, testato e riconfermato in
  almeno 5 round diversi, con SOAP di ricchezza diversa, con batch diversi, con
  `element_agnostic` sia vero che falso. Nessuna eccezione trovata.

---

## Atto 2 — Togliere l'identità chimica: il crollo e la lunga risalita

La domanda successiva è se la geometria pura (senza sapere *quali* elementi chimici
compongono la struttura, `element_agnostic: true`) basti da sola a distinguere le famiglie.
La risposta iniziale è un crollo drastico: accuratezza 0.97-0.99 → **0.45-0.71**.

La causa, misurata e non ipotizzata, è puramente dimensionale: attivare
`element_agnostic` fa collassare il descrittore SOAP da 12960 a 40 dimensioni (324 volte
più piccolo), perché tutte le specie chimiche vengono fuse in un solo canale. Non è un
problema "qualitativo" di rumore chimico tolto — è un problema di quanti numeri il
descrittore ha ancora a disposizione.

Da qui parte una ricerca sistematica delle leve giuste per richiudere il divario:

- **Arricchire SOAP** (aumentare `n_max`/`l_max`, dimensione 40→252→495) **aiuta in modo
  monotono ma con rendimenti fortemente decrescenti**: il salto 40→252 vale +0.2/+0.4 punti
  di accuratezza, il salto successivo 252→495 vale solo +0.02/+0.03.
- **Scalare il dataset** (raddoppiare il numero di strutture, SOAP invariato) **non aiuta
  — anzi peggiora leggermente** l'accuratezza media, a parità di budget di training (le
  stesse epoche/early-stopping "tagliano corto" rispetto a un dataset più grande).
  Ridistribuire i dati per spacegroup invece che per famiglia (con uno sbilanciamento
  34× tra la classe più piccola e la più grande) **non cambia questa conclusione**: ancora
  nessun guadagno netto.
- **Il budget di training** (batch 32→128, epoche 100→200, pazienza dell'early stopping
  20 invece di 10) **è la leva singola più efficace trovata in tutto lo studio**: il
  miglior run raggiunge **0.9726** *(round 7)* — a soli 2.1 punti dal riferimento storico
  del round 1 (0.9938), partendo da un dataset "solo" 6 volte più piccolo in dimensione
  SOAP. Questa configurazione (supcon, cosine, τ=0.05, encoder `[128,64]`, batching random,
  batch 128/epoche 200, SOAP dim=252, n_species=1) diventa **la configurazione di
  riferimento per tutto il resto dello studio**, sotto `element_agnostic: true`.

La lezione di questo atto, in una frase: **il collo di bottiglia non era la scala del
dataset, era il descrittore e il budget di training** — due leve facili da confondere con
"serve più roba" ma di natura completamente diversa.

Nota a margine, esplorata più avanti e poi abbandonata: si è provato a rendere l'intera
pipeline invariante alla scala assoluta della cella (`soap.normalize_distances: true`),
sperando in una maggiore robustezza. Quattro tentativi successivi (normalizzazione ingenua,
raggio di cutoff riproporzionato, jitter relativo, cutoff relativo più ricco) non hanno mai
richiuso il divario con la configurazione di riferimento (il migliore resta ~8-13 punti
sotto). L'ipotesi più solida: la scala assoluta della cella porta essa stessa un segnale
correlato alla famiglia (le celle Cubic sono in media ~20% più grandi, probabile artefatto
del generatore pyxtal) — un descrittore davvero scala-invariante butta via quel segnale per
costruzione. Non chiuso, solo abbandonato con una spiegazione plausibile.

---

## Atto 3 — Il giallo architetturale: perché una vecchia run era così più bella

Durante lo studio riemerge una run storica pre-progetto (ad agosto) la cui visualizzazione
2D nativa di supcon era sorprendentemente buona (silhouette 0.73), molto meglio di
qualunque cosa vista fin qui nei round "ufficiali". Indagine in stile detective:

1. Si riproduce la vecchia run con il **codice storico** e la stessa config → qualità
   confermata (silhouette 0.666) — non era un fluke.
2. Si prende la stessa identica config ma la si allena col **codice attuale** → quasi
   collasso totale (silhouette -0.213).
3. Si sospetta l'ottimizzatore (VeLO vs Adam) e si forza VeLO → **ancora peggio**
   (-0.291): non era quello.

La causa reale, isolata per bisezione nella storia del codice, è un singolo commit che ha
introdotto la `ProjectionTail`: la loss contrastiva, prima calcolata direttamente
sull'embedding del corpo, ora viene calcolata su uno spazio proiettato separato. Scelta
**architetturalmente corretta** per l'accuratezza di classificazione a valle (è la
convenzione SimCLR/Khosla), ma con un effetto collaterale: quando il corpo ha una
dimensione molto piccola (2, come nella vecchia run), il corpo stesso è libero di
collassare verso un punto fisso, perché nulla lo vincola più direttamente — la loss è
comunque soddisfatta tramite la proiezione. Questo spiega, in modo sistemico e non
episodico, perché la qualità 2D *nativa* di supcon è sempre stata deludente in tutto
l'Atto 1: non è un problema di iperparametri round per round, è un problema strutturale
del design a due spazi.

La soluzione pratica, trovata senza toccare una riga di codice: allenare una tail di
visualizzazione **più ricca** sul corpo già congelato — modalità `family_and_spacegroup`
invece di `family_only`, 200 epoche invece di 40, batch 128 invece di 32, un MLP più
profondo `[32,16]`. Risultato sul corpo del round 1 (accuratezza di classificazione
invariata, 0.9938): silhouette 0.660, k-means ARI 0.919, k-NN accuracy 0.975 — **eguaglia
o supera vae**, mantenendo il vantaggio di accuratezza di supcon. Questa "tail ricca"
diventa la ricetta standard, riapplicata più volte nel resto dello studio (ogni volta con
lo stesso effetto: enorme salto di qualità 2D, zero costo in accuratezza).

Un'idea nata da questa scoperta e poi testata esplicitamente — far sì che la tail di
classificazione/visualizzazione **sostituisca** del tutto la `ProjectionTail` della fase 1,
invece di restare un add-on di fase 2 — è stata provata su un branch separato e **respinta**:
peggiora ogni metrica (accuratezza famiglia 0.9730→0.9246, qualità 2D dell'embedding del
corpo anch'essa peggiore). La spiegazione: lo spazio di proiezione ampio e "usa e getta"
dà alla loss contrastiva libertà di organizzarsi senza costringere il corpo stretto nella
stessa geometria vincolata — un vantaggio del design esistente, non solo convenzione
storica.

---

## Atto 4 — Alzare l'asticella: dallo spacegroup al perché una famiglia non impara mai

Con la famiglia ormai "risolta" (riferimento 0.9726-0.9938 a seconda del setup), lo studio
si sposta sull'obiettivo molto più difficile: classificare lo **spacegroup** (decine di
classi per famiglia, non più solo 7 famiglie). Si introduce un'architettura gerarchica a
due stadi: un router di famiglia (il corpo già allenato) seguito da un "esperto" per
spacegroup dedicato a ciascuna famiglia.

Tre iterazioni progressive sull'input dato agli esperti mostrano quanto conti la
rappresentazione di partenza, non solo la capacità del classificatore:
- Esperti sull'embedding compresso del corpo (8 dimensioni): accuratezza oracle **0.25** —
  debole ma sempre ben sopra il caso, tranne Triclinic.
- Esperti sul descrittore SOAP nativo (252 dimensioni) invece che sull'embedding
  compresso: **0.57** (+32 punti) — il collo di bottiglia a 8 dimensioni, non SOAP stesso,
  era il vero limite.
- Esperti con capacità e budget di training "alla pari del corpo" (`[128,64]`, batch
  128/epoche 200): **0.93** — sei famiglie su sette superano il 90%.

E qui emerge, per la prima volta, un'anomalia che diventerà il filo conduttore del resto
della storia: **una famiglia — Cubic — resta inchiodata al 50%**, isolata da tutte le
altre per un margine enorme, mentre le altre sei arrivano tutte oltre lo 0.89.

In parallelo si esplora una variante "simmetrica": invece di un classificatore diretto per
spacegroup, un piccolo SupCon dedicato per famiglia (stessa filosofia contrastiva del
corpo principale, applicata a livello di spacegroup). Il compromesso è netto: la
classificazione è peggiore (oracle 0.85 contro 0.93 del classificatore diretto,
specialmente su Cubic), ma la **visualizzazione 2D è migliore** su quasi ogni famiglia —
una rappresentazione allenata in modo contrastivo si presta meglio a una proiezione 2D di
quanto non faccia lo strato nascosto di un classificatore softmax. Allargare la dimensione
del collo di bottiglia di questo SupCon-SG (8→16→32) recupera quasi tutto il divario di
accuratezza (0.85→0.90), **ma non su Cubic**, che resta fermo al 43-48% qualunque leva di
capacità venga tirata — seconda conferma indipendente che il problema di Cubic non è "poca
capacità del modello".

Una domanda architetturale collaterale, testata con rigore: la convenzione classica
(Khosla et al. 2020) vuole un encoder largo e una proiezione stretta. Qui succede
l'opposto (round 7: encoder 8-dim, proiezione 128-dim) e **funziona meglio**. Un test
sistematico lo conferma: restringere la proiezione sotto l'encoder peggiora l'accuratezza
in modo monotono, fino quasi al collasso (0.8528 → 0.5587 nella variante più estrema);
eliminare del tutto la proiezione (loss diretta sull'encoder) costa comunque accuratezza a
ogni dimensione provata. **Più spazio aiuta sempre, ovunque lo si metta** — l'intuizione
"alla Khosla" si rivela sbagliata per questo problema. Combinando tutte le leve vincenti
trovate fin qui (encoder allargato `[256,128]`, collo di bottiglia 32, proiezione 128,
teste di classificazione/visualizzazione allargate) si ottiene la miglior configurazione
"simmetrica" mai raggiunta per lo spacegroup: oracle **0.9223**, a poco più di un punto dal
classificatore diretto (0.9322) — il divario più piccolo mai raggiunto in questa linea di
lavoro. Un ultimo giro di tuning, questa volta solo sul visualizzatore per-famiglia (a
corpo fisso), mostra che ogni famiglia (tranne Cubic) ha il proprio ottimo locale di
architettura — più profondità aiuta quasi ovunque, più larghezza no, e i due insieme non
aiutano ulteriormente: un pattern diverso da "più capacità aiuta sempre" visto fin qui.

---

## Atto 5 — Il finding centrale: perché Cubic non può funzionare (e non è un bug)

Cubic è rimasta, in ogni singolo esperimento dell'Atto 4, la famiglia più difficile — e
l'unica a non rispondere a **nessuna** variazione di capacità (encoder più largo, collo di
bottiglia più ampio, teste più capaci). Questo round chiude la questione con tre misure
indipendenti, quasi tutte model-free.

**Misura 1 — i gradi di libertà geometrici reali, per famiglia.** Calcolati direttamente
dalle strutture generate (deviazione standard dei rapporti c/a, b/a, e degli angoli di
cella):

| famiglia | gradi di libertà della forma della cella |
|---|---:|
| Cubic | **0** (a=b=c, tutti gli angoli a 90° — per definizione) |
| Esagonale / Tetragonale / Trigonale | 1 |
| Ortorombica | 2 |
| Monoclina | 3 |
| Triclina | 5 |

Non un'approssimazione: è un fatto cristallografico esatto.

**Misura 2 — un effetto soglia, non un gradiente (la scoperta chiave).** Confrontando i
gradi di libertà con l'accuratezza oracle sul miglior corpo disponibile:

| gradi di libertà | famiglie | accuratezza |
|---:|---|---:|
| 0 | Cubic | **0.497** |
| 1 | Tetragonale / Trigonale / Esagonale | 0.990 / 0.991 / 0.992 |
| 2 | Ortorombica | 0.990 |
| 3 | Monoclina | 0.998 |
| 5 | Triclina | 0.999 |

**Ogni famiglia con almeno un grado di libertà sta sopra 0.989, indipendentemente da
quanti gradi di libertà ne abbia in più** (1, 2, 3 o 5 non fanno differenza tra loro).
Solo lo zero assoluto (Cubic) fa collassare l'accuratezza, di oltre 49 punti. Non è una
relazione graduale "più forma libera = più accuratezza" — è un gradino netto. Questo
spiega perfettamente perché nessun intervento di capacità negli atti precedenti abbia mai
smosso Cubic: il problema non è "un segnale debole che serve più modello per sfruttare" —
è l'**assenza totale** di un segnale macroscopico di forma per un descrittore
intrinsecamente locale come SOAP. Basta un solo parametro di forma libero perché la
geometria locale più quell'unico indizio globale bastino quasi a risolvere il problema;
senza, resta solo la disposizione fine degli atomi (posizioni di Wyckoff), un segnale
molto più debole e locale, che nessuna quantità di capacità aggiuntiva può sostituire con
un segnale macroscopico che semplicemente non è nel descrittore.

**Misura 3 — la confusione è diffusa, non concentrata (esclude l'ipotesi della
chiralità).** Sulla matrice di confusione di Cubic (2500 punti, 36 spacegroup, 1257
errori): 203 coppie diverse di spacegroup confuse (su un massimo teorico di 1260), 34 dei
36 spacegroup coinvolti in almeno un errore, le 5 coppie più confuse coprono solo l'8.3%
degli errori totali. Il pattern opposto a quello atteso se poche coppie quasi-degeneri
(es. coppie enantiomorfe, l'ipotesi iniziale di qualche atto fa) fossero la causa — quello
mostrerebbe pochi "punti caldi" dominanti. Confronto visivo diretto: Tetragonale (68
classi, il numero più alto di tutte le famiglie) mostra una diagonale quasi perfetta nella
sua matrice di confusione, **smentendo direttamente** l'ipotesi alternativa "più classi =
più difficile".

**Conclusione**: il tetto di ~50% di Cubic è un limite strutturale genuino di un
descrittore geometrico locale (SOAP) applicato a un sistema cristallino a zero gradi di
libertà nella forma della cella — verificato con tre misure indipendenti, in gran parte
indipendenti dal modello. Non è un problema di training, di capacità, di volume di dati, o
di architettura. Le uniche strade plausibili per smuoverlo sono qualitativamente diverse:
descrittori che catturano segnale a più lungo raggio, o feature esplicitamente derivate
dalla simmetria dello spacegroup — non descrittori locali come SOAP. Il finding più forte,
rigoroso e completo di tutto lo studio.

---

## Epilogo — un promemoria metodologico

Un'osservazione trasversale, utile per interpretare correttamente tutti i numeri sopra: si
è misurata una **variabilità run-to-run di circa ±0.02 di accuratezza**, anche in
configurazioni che in teoria non dovrebbero cambiare nulla nel percorso di
classificazione (non-determinismo GPU/XLA su 200 epoche di training). Qualunque differenza
più piccola di questo margine, in questa storia, va letta con cautela — non tutte le
piccole vittorie riportate nei singoli round sono statisticamente solide, e questo è
esplicitamente riconosciuto nelle note originali.

Per le domande ancora aperte e i prossimi passi, vedi `05_domande_aperte.md`. Per un
indice di dove trovare le figure reali già pronte per ogni scena di questa storia, vedi
`04_figure_disponibili.md` e (più dettagliato) `experiments/plots/README.md`.
