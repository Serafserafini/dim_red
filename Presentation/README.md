# Materiale per la presentazione — dim_red

Cartella di **sole note**, per preparare una presentazione sul progetto `dim_red` e sullo
studio "family_only" condotto al suo interno. Nessun file di codice è stato toccato per
produrre questo materiale — solo lettura di `CLAUDE.md`, `docs/`, `experiments/*_notes.md`,
`experiments/milestones/`, `experiments/plots/` e dei config in `configs/`. Il codice
continua a essere usato/modificato in parallelo fuori da questa cartella: questi file sono
uno snapshot ragionato, non un tracciamento live dei run.

## Punto di partenza: `struttura_talk.md`

**Comincia da lì.** È la scaletta del talk ricostruita 1:1 dal brainstorm
(`/home/seraftu/Work/Presentations/Ott26/Brainstorm.pdf`): motivazione (nested
sampling/brass) → problema (simmetrie rumorose) → lavoro precedente (RDF/Steinhardt +
UMAP) → la domanda che porta a `dim_red` → introduzione tecnica (SOAP, encoder+classifier,
perché non AE/VAE) → architettura SupCon a 4 blocchi → **Risultati** (riempita con i
numeri reali dello studio) → **Futuro** (embedding MACE). Le sezioni 1-4 riportano
fedelmente il brainstorm; la sezione Risultati è quella arricchita con il lavoro fatto qui.

## Gli altri file

- **`storia_studio.md`** — il racconto completo e disteso dei risultati dello studio
  "family_only" (round 1→17 riorganizzati in ordine logico, non cronologico), con tutti i
  numeri e le tabelle. `struttura_talk.md` ne cita solo lo scheletro — questo è il
  materiale di riferimento per scrivere gli speaker notes della sezione Risultati.
- **`01_overview_progetto.md`** — cos'è `dim_red`, la pipeline, i 5 `model_kind` a
  confronto. Materiale architetturale/didattico, utile per la sezione 6
  (SupCon a 4 blocchi) e per rispondere a domande tecniche del pubblico.
- **`04_figure_disponibili.md`** — indice di dove si trovano le figure reali già pronte
  (nessuna richiede un nuovo run), organizzate per sezione del talk.
- **`05_domande_aperte.md`** — altri open thread oltre a MACE, per la chiusura.

## Nota su un documento preesistente

Esiste già `presentation_notes.md` in root (non in questa cartella, quindi non toccato) —
una bozza di talk in **inglese**, ~30 minuti, con un taglio diverso: una "scala di
complessità" didattica sui 6-7 metodi (PCA → UMAP → AE → VAE → SupCon → CGCNN → MACE), come
tour del toolkit più che come racconto dei risultati. Scritta l'8 settembre 2026, **prima**
dei round 8-17 e prima del brainstorm — non è la base di questo materiale, ma può tornare
utile come riferimento sui singoli metodi (PCA/UMAP/AE/VAE) se il talk finale vuole
accennarci, dato che lo studio reale ha usato quasi solo `supcon`.

## Lingua

Tutte le note qui sono in italiano, come tutte le note di ricerca originali in
`experiments/` e il brainstorm stesso. Se la presentazione finale deve essere in inglese,
va tradotta a valle — ditemelo e lo faccio.
