# dim_red

Toolkit per ridurre la dimensionalità di strutture cristalline e allenare rappresentazioni latenti che separano famiglie cristalline e gruppi spaziali. Layout `src/`, test con `pytest`, documentazione Sphinx in `docs/`.

## Cosa contiene

- **Dati**: `fetch.py` (Materials Project), `generate.py` (strutture simmetriche sintetiche con `pyxtal`), `augmentation.py` (jitter posizionale, vacanze, supercelle).
- **Featurizzazione**: `soap.py` (SOAP via `dscribe`), `cgcnn/graph.py` (grafo di legami), `mace/` (embedding di un MACE pre-addestrato e congelato).
- **Riduzione classica**: `pca.py`, `umap.py`, `utils.py` (standardizzazione).
- **Modelli** (`model_kind`): `supcon` (Supervised Contrastive su SOAP), `supcon_mace` (stessa ricetta sugli embedding MACE), `cgcnn` (rete a grafo con classificazione congiunta).
- **Fase 2 (tail)**: `classification`, `visualization`, `hierarchical_supcon` (esperti per famiglia) sopra un corpo congelato.
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
| `dimred-run <config.yaml>` | una run singola (dataset → modello → artefatti) |
| `dimred-sweep <config.yaml>` | grid search |
| `dimred-rerun <run_dir>` | rilancia una run salvata |
| `dimred-compare <dir>` | grafici di confronto di una sweep |
| `dimred-train-tail <config.yaml>` | allena un tail di fase 2 su una run |
| `dimred-apply <structures.extxyz> <run_dir>` | applica una run a nuove strutture |
| `dimred-benchmark <runs>... --output <csv>` | tabella di qualità tra run/model_kind |

Config di esempio in `configs/`, riferimento completo dei campi in `examples/config_reference.example.yaml`, descrizione di ogni blocco in `docs/` (`sphinx-build docs docs/_build`).

## Test

```bash
pytest tests/test_pca.py        # un file
pytest                          # tutta la suite
```
