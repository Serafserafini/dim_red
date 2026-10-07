# Confronto & benchmark

Due strumenti distinti, entrambi progettati per **non** classificare o scegliere automaticamente un "run migliore" — producono solo tabelle/grafici che l'utente legge e giudica da sé.

```{list-table}
:header-rows: 1
:widths: 25 75

* - Strumento
  - Ambito
* - `pipeline.compare`
  - Per **una singola cartella di sweep** di run FullStack: una suite di grafici (curve di loss, metrica finale vs. iperparametro, griglia dello spazio latente, istogramma spacegroup, confronto accuratezza) **per ogni stack**, in `<sweep>/comparison/<stack>/`. Serve `--heads-name` se uno stack ha più set di heads.
* - `pipeline.benchmark`
  - Una singola tabella CSV larga, **una riga per stack** di ogni run FullStack (un run ha uno stack `family` e fino a 7 esperti). Accetta un mix di directory di run singoli e directory di sweep in un'unica chiamata; `--heads-name` se ci sono più set di heads.
```

```{eval-rst}
.. autofunction:: dim_red.pipeline.compare.generate_comparison_report

.. autofunction:: dim_red.pipeline.benchmark.generate_benchmark_table

.. autofunction:: dim_red.pipeline.benchmark.generate_benchmark_plots
```

## Le metriche di qualità dell'embedding

Calcolate da `dim_red.analysis.metrics.embedding_quality_metrics` (solo scikit-learn), sia sull'embedding nativo (qualunque sia `latent_dim`) sia, con prefisso `_2d`, su una proiezione 2D scelta in ordine di preferenza: la visualization head già allenata con `viz_dim=2` → l'embedding nativo se già 2D → una proiezione PCA deterministica come ultima risorsa.

```{eval-rst}
.. autofunction:: dim_red.analysis.metrics.embedding_quality_metrics
```

## Inferenza — applicare un run a nuove strutture

`dimred-apply <structures.extxyz> <run_dir>`. Per un run FullStack (`--heads-name` se ci sono più set di heads) scrive in `<run_dir>/applied` un CSV di predizioni (`family`, `expert`, `spacegroup` con le probabilità; vuoti se l'esperto non è allenato) e un npz delle coordinate di visualizzazione. Dalla libreria: `FullStack.open(run).predict(structures)`, vedi {doc}`fullstack`.

```{eval-rst}
.. autofunction:: dim_red.pipeline.full_stack.apply_to_structures
```

Per un run cgcnn l'applicazione proietta le strutture nello spazio latente del run (recupera `feature_mean`/`feature_std` da `embeddings.npz`) e le disegna insieme al dataset originale:

```{eval-rst}
.. autofunction:: dim_red.pipeline.inference.load_trained_run

.. autofunction:: dim_red.pipeline.inference.apply_model_to_structures
```
