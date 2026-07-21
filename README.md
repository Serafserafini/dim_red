# dim_red - Codebase & Suite di Test

Codebase Python per l'implementazione e la valutazione di algoritmi di Riduzione della Dimensionalità (Dimensionality Reduction), strutturata secondo le best practice industriali con layout `src/` e test automatizzati con `pytest`.

## 📁 Struttura del Progetto

```text
dim_red/
├── .gitignore             # File ignorati da Git
├── pyproject.toml         # Configurazione del pacchetto e di pytest
├── requirements.txt       # Dipendenze di progetto
├── README.md              # Documentazione
├── src/                   # Codice sorgente del pacchetto
│   └── dim_red/
│       ├── __init__.py    # Inizializzazione pacchetto
│       ├── pca.py         # Implementazione dell'algoritmo PCA
│       └── utils.py       # Funzioni utility (es. standardizzazione)
└── tests/                 # Suite dei test unitari e d'integrazione
    ├── __init__.py
    ├── conftest.py        # Fixture e setup di test per pytest
    ├── test_pca.py        # Test per PCA
    └── test_utils.py      # Test per le utility
```

## 🚀 Installazione e Setup

### 1. Creazione ambiente virtuale (consigliato)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Installazione dipendenze

```bash
pip install -r requirements.txt
```

In alternativa, per installare il pacchetto in modalità editabile con dipendenze di sviluppo:

```bash
pip install -e .[dev]
```

## 🧪 Esecuzione dei Test

Per eseguire tutti i test unitari presenti nella cartella `tests/`:

```bash
pytest
```

Per eseguire i test in modalità prolissa (verbose):

```bash
pytest -v
```
