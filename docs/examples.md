# Catalogo config di esempio

I YAML in `configs/` (alcuni job SLURM in `slurm/`) mostrano combinazioni reali di quanto documentato nelle pagine precedenti. `tests/test_example_configs.py` carica ogni config shipped: un esempio che non carica rompe il test.

## FullStack (supcon / supcon_mace)

```{list-table}
:header-rows: 1
:widths: 40 60

* - File
  - Cosa mostra
* - `full_stack.example.yaml`
  - run di prova: uno stack `family` e due esperti (`cubic`, `tetragonal` con override), dataset pyxtal piccoli. `dimred-run`.
* - `full_stack_best_combo.example.yaml`
  - la combinazione di riferimento (tau 0.05 cosine, encoder `[256, 128]`, latente 32, proiezione 128, 200 epoche con early stopping).
* - `full_stack_sweep.example.yaml`
  - grid search su un config FullStack (`base` + `grid` con percorsi puntati). `dimred-sweep`, poi `dimred-compare`/`dimred-benchmark`.
* - `train_heads.example.yaml`
  - nuovo set di heads su body già allenati. `dimred-train-heads <config> <run_dir> --heads-name NAME`.
```

## cgcnn

```{list-table}
:header-rows: 1
:widths: 40 60

* - File
  - Cosa mostra
* - `single_run_cgcnn.example.yaml`
  - pyxtal + cgcnn (grafo, no SOAP). `dimred-run`.
* - `tail_train_classification.example.yaml`
  - `TailTrainConfig` per un tail di classificazione su un run cgcnn. `dimred-train-tail`.
* - `tail_train_visualization.example.yaml`
  - `TailTrainConfig` per un tail di visualizzazione su un run cgcnn.
```

## Riferimento dei campi

Per lo schema FullStack i riferimenti sono le config `configs/full_stack*.example.yaml` e {doc}`fullstack`; per `RunConfig` cgcnn le pagine {doc}`runconfig`, {doc}`featurization` e {doc}`aux_supcon`.
