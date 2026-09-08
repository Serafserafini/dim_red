# Catalogo config di esempio

18 YAML in `configs/` (più le loro copie in `slurm/` per il lancio su cluster) mostrano combinazioni reali di quanto documentato nelle pagine precedenti.

## Run singoli

```{list-table}
:header-rows: 1
:widths: 40 60

* - File
  - Cosa mostra
* - `single_run.example.yaml`
  - fetch + vae, la baseline più semplice.
* - `single_run_pyxtal.example.yaml`
  - pyxtal + vae, dataset sintetico.
* - `single_run_autoencoder.example.yaml`
  - fetch + autoencoder deterministico.
* - `single_run_supcon.example.yaml`
  - pyxtal + supcon, con blocco `tails` commentato come riferimento.
* - `single_run_supcon_hierarchical.example.yaml`
  - supcon + tail gerarchico attivo.
* - `single_run_cgcnn.example.yaml`
  - pyxtal + cgcnn (grafo, no SOAP).
* - `single_run_mace.example.yaml`
  - corpo mace congelato + `tails` per la classificazione.
```

## Sweep (grid search)

```{list-table}
:header-rows: 1
:widths: 40 60

* - File
  - Cosa mostra
* - `sweep.example.yaml`
  - sweep generico di riferimento.
* - `sweep_supcon.example.yaml`
  - sweep sugli iperparametri di supcon.
* - `sweep_aux_heads.example.yaml`
  - sweep sulle head ausiliarie (vae/autoencoder).
* - `sweep_cgcnn.example.yaml`
  - sweep sugli iperparametri di cgcnn.
```

## Tuning & benchmark incrociato

```{list-table}
:header-rows: 1
:widths: 40 60

* - File
  - Cosa mostra
* - `tuning_sweep_{vae,autoencoder,supcon,cgcnn}.example.yaml`
  - tuning interno a ciascun `model_kind`, da abbinare a `benchmark_pyxtal_base`.
* - `benchmark_pyxtal_base.example.yaml`
  - config base condivisa per il flusso a due stadi tuning → benchmark tra `model_kind`.
```

## Fase 2 standalone

```{list-table}
:header-rows: 1
:widths: 40 60

* - File
  - Cosa mostra
* - `tail_train_classification.example.yaml`
  - `TailTrainConfig` standalone per un tail di classificazione.
* - `tail_train_visualization.example.yaml`
  - `TailTrainConfig` standalone per un tail di visualizzazione.
```

## Riferimento completo

`examples/config_reference.example.yaml` elenca ogni campo con default/commenti — non pensato per essere lanciato così com'è, solo come promemoria dell'intera forma dello schema YAML.
