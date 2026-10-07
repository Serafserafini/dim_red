# dim_red

Toolkit per ridurre la dimensionalità di strutture cristalline e allenare rappresentazioni latenti che separano famiglie cristalline e gruppi spaziali. Layout `src/`, test con `pytest`, documentazione Sphinx in `docs/`.

## Cosa contiene

- **Dati**: `fetch.py` (Materials Project), `generate.py` (strutture simmetriche sintetiche con `pyxtal`), `augmentation.py` (jitter posizionale, vacanze, supercelle).
- **Featurizzazione**: `soap.py` (SOAP via `dscribe`), `cgcnn/graph.py` (grafo di legami), `mace/` (embedding di un MACE pre-addestrato e congelato).
- **Riduzione classica**: `pca.py`, `umap.py`, `utils.py` (standardizzazione).
- **Modelli** (`model_kind`): `supcon` (Supervised Contrastive su SOAP) e `supcon_mace` (stessa ricetta sugli embedding MACE), allenati solo tramite `FullStack` (uno stack per sistema cristallino + fino a 7 esperti per spacegroup, ognuno con classificatore e visualizzatore); `cgcnn` (rete a grafo con classificazione congiunta).
- **`pipeline/`**: orchestrazione guidata da YAML (`RunConfig`), sweep, confronto, benchmark, inferenza su nuove strutture.
- **`analysis/`**: grafici e metriche di qualità degli embedding.

## Installazione

```bash
pip install -e ".[dev]"      # soap, fetch, umap, train (jax), pyxtal, analysis, pipeline + pytest
pip install -e ".[mace]"     # opzionale, solo per model_kind: supcon_mace
```

Il progetto è sviluppato nell'ambiente conda `dmred`.

## Uso

Comandi installati (`pyproject.toml`):

| Comando | Cosa fa |
|---|---|
| `dimred-run <config.yaml>` | una run (FullStack o cgcnn, in base alle chiavi della config) |
| `dimred-sweep <config.yaml>` | grid search su una config FullStack |
| `dimred-rerun <run_dir>` | rilancia una run salvata |
| `dimred-train-heads <config.yaml> <run_dir> --heads-name NAME` | nuovo set di heads (classificatore + viz) su body FullStack già allenati |
| `dimred-compare <dir>` | grafici di confronto di una sweep, una suite per stack |
| `dimred-train-tail <config.yaml> <run_dir>` | tail di fase 2 su un run cgcnn |
| `dimred-apply <structures.extxyz> <run_dir>` | applica una run a nuove strutture |
| `dimred-benchmark <runs>... --output <csv>` | tabella di qualità, una riga per stack |

Config di esempio in `configs/`, partire da `configs/full_stack.example.yaml`; descrizione di ogni blocco in `docs/` (`docs/fullstack.md` per supcon/supcon_mace) (`sphinx-build docs docs/_build`).

## Test

```bash
pytest tests/test_pca.py        # un file
pytest                          # tutta la suite
```
