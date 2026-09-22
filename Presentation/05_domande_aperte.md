# Domande aperte / next steps

Per la chiusura del talk, oltre all'idea MACE già in `struttura_talk.md` (sezione 8,
esplicitamente nel brainstorm). Fonte: note dei round, sintetizzate.

- **Il tetto di Cubic (~0.50) è accettato come limite strutturale, non "risolto"** — la
  strada indicata dal round 17 stesso è qualitativamente diversa da tutto ciò che è stato
  provato finora: servirebbe un descrittore a raggio più lungo del locale SOAP, o feature
  esplicitamente derivate dalla simmetria dello spacegroup. Non ancora tentato. **MACE
  (sezione 8) è il candidato più naturale**, dato che cattura interazioni a più corpi non
  puramente locali.
- **La scale-invariance (`soap.normalize_distances`) è stata abbandonata, non risolta**:
  quattro tentativi (round 9-12) non hanno mai richiuso il divario di accuratezza (~8-13
  punti nel migliore dei casi) rispetto alla configurazione di riferimento. Se in futuro
  serve generalizzare a strutture reali (non sintetiche pyxtal) con scale di cella molto
  variabili, resta un problema aperto.
- **Il SupCon-SG "simmetrico" (contrastivo, stessa filosofia del corpo principale) resta
  indietro di ~1 punto rispetto al classificatore diretto per spacegroup** (0.9223 vs
  0.9322 oracle) — mai del tutto richiuso. Ha però il vantaggio di una visualizzazione 2D
  migliore su quasi ogni famiglia: è un trade-off, non deciso automaticamente (per scelta
  di design del progetto — mai una selezione automatica di "vincitore").
- **Nessun run ha ancora combinato tutte le leve vincenti in un'unica configurazione
  "definitiva"** su famiglia, spacegroup e visualizzazione insieme — il `best_combo` del
  round 15 è il tentativo più vicino, ma esplicitamente non vince su tutti e tre gli assi
  contemporaneamente (accuratezza famiglia leggermente sotto il record ottenuto allargando
  solo l'encoder).
