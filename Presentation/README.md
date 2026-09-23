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
perché non AE/VAE) → architettura SupCon a 4 blocchi → **Risultati** (leggera, l'idea
illustrata non un report) → **Futuro** (embedding MACE). Le sezioni 1-4 riportano
fedelmente il brainstorm; la sezione Risultati è arricchita col lavoro fatto qui, ma tenuta
apposta breve.

**Vincoli del talk** (vedi il file per il dettaglio): ~20 minuti, niente racconto
round-per-round degli esperimenti, niente costi di training, framing onesto
"proof-of-concept, non ancora nulla di perfettamente funzionante", e su Cubic solo un
accenno cauto (nessuna spiegazione di causa-radice — non ancora abbastanza sicura da
portare in pubblico).

## I file operativi per preparare il talk vero e proprio

- **`discorso.md`** — il discorso parola per parola, sezione per sezione, sincronizzato con
  `struttura_talk.md`. Rifinito insieme, 14 slide, timestamp ricalcolati sul conteggio
  parole reale (~18-19 minuti totali).
- **`brief_slides_claude_design.md`** — brief sintetico slide-per-slide (titoli, bullet,
  indicazioni di visual), sincronizzato 1:1 con `discorso.md`.
- **`prompt_per_claude_design.md`** — **il file da usare per generare davvero il deck**:
  un prompt autosufficiente (include tutto il contenuto di discorso + brief, incluse le
  speaker notes) da incollare in Claude Design per ottenere un `.pptx` editabile. Contiene
  anche una checklist di quali immagini caricare prima di lanciarlo.

## Gli altri file

- **`storia_studio.md`** — il racconto completo e disteso dei risultati dello studio
  "family_only" (round 1→17 riorganizzati in ordine logico, non cronologico), con tutti i
  numeri e le tabelle (incluso il dettaglio completo sul finding di Cubic, round 17).
  **Non è più la base della sezione Risultati** del talk (che va tenuta leggera, vedi
  `struttura_talk.md`) — resta come materiale di consultazione per dopo il talk, o per te
  stesso se in futuro il finding su Cubic ti convince abbastanza da volerlo raccontare.
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
