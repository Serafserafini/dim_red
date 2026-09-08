# Confronto & benchmark

Due strumenti distinti, entrambi progettati per **non** classificare o scegliere automaticamente un "run migliore" — producono solo tabelle/grafici che l'utente legge e giudica da sé.

```{list-table}
:header-rows: 1
:widths: 25 75

* - Strumento
  - Ambito
* - `pipeline.compare`
  - Un pannello di grafici (curve di loss, metrica finale vs. iperparametro, griglia dello spazio latente, istogramma spacegroup/famiglia, confronto accuratezza head ausiliarie) per **una singola cartella di sweep**, assumendo colonne di `loss_history.csv` comparabili al suo interno.
* - `pipeline.benchmark`
  - Una singola tabella CSV larga che confronta **`model_kind` diversi insieme** (cosa che `compare` non fa, perché le colonne/scale della loss divergono per kind). Accetta un mix di directory di run singoli e directory di sweep in un'unica chiamata.
```

```{eval-rst}
.. autofunction:: dim_red.pipeline.compare.generate_comparison_report

.. autofunction:: dim_red.pipeline.benchmark.generate_benchmark_table

.. autofunction:: dim_red.pipeline.benchmark.generate_benchmark_plots
```

## Le metriche di qualità dell'embedding

Calcolate da `dim_red.analysis.metrics.embedding_quality_metrics` (solo scikit-learn), sia sull'embedding nativo (qualunque sia `latent_dim`) sia, con prefisso `_2d`, su una proiezione 2D scelta in ordine di preferenza: un tail di visualizzazione già allenato con `viz_dim=2` → l'embedding nativo se già 2D → una proiezione PCA deterministica come ultima risorsa.

```{eval-rst}
.. autofunction:: dim_red.analysis.metrics.embedding_quality_metrics
```

## Inferenza — applicare un run a nuove strutture

`dimred-apply <structures.extxyz> <run_dir>`: carica un run già completato e proietta strutture mai viste nel suo spazio latente, senza ricalcolare SOAP sul training set (recupera `feature_mean`/`feature_std` direttamente da `embeddings.npz`, con un fallback solo per run salvati prima che questo campo esistesse).

```{eval-rst}
.. autofunction:: dim_red.pipeline.inference.load_trained_run

.. autofunction:: dim_red.pipeline.inference.apply_model_to_structures
```
