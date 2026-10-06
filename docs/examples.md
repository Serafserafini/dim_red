# Catalogo config di esempio

I YAML in `configs/` (più le loro copie in `slurm/` per il lancio su cluster) mostrano combinazioni reali di quanto documentato nelle pagine precedenti.

## Run singoli

```{list-table}
:header-rows: 1
:widths: 40 60

* - File
  - Cosa mostra
* - `single_run_supcon.example.yaml`
  - pyxtal + supcon, con blocco `tails` commentato come riferimento.
* - `single_run_pyxtal.example.yaml`
  - il più semplice: dataset sintetico pyxtal + supcon con i default.
* - `single_run_supcon_hierarchical.example.yaml`
  - supcon + tail di famiglia e visualizzazione automatici; gli esperti per famiglia si aggiungono con `dimred-train-tail`.
* - `single_run_supcon_best_combo.example.yaml`
  - la combinazione di riferimento (encoder `[256, 128]`, latente 32, proiezione 128).
* - `single_run_supcon_encoder_256_128.example.yaml`
  - come sopra con l'encoder più largo, base storica di `best_combo`.
* - `single_run_cgcnn.example.yaml`
  - pyxtal + cgcnn (grafo, no SOAP).
```

## Sweep (grid search)

```{list-table}
:header-rows: 1
:widths: 40 60

* - File
  - Cosa mostra
* - `sweep_supcon.example.yaml`
  - sweep sugli iperparametri di supcon.
* - `sweep_cgcnn.example.yaml`
  - sweep sugli iperparametri di cgcnn.
* - `tuning_sweep_supcon_family_only_round7.example.yaml`
  - la sweep del "round 7", config di riferimento per la famiglia con supcon.
```

## Tuning & benchmark incrociato

```{list-table}
:header-rows: 1
:widths: 40 60

* - File
  - Cosa mostra
* - `tuning_sweep_{supcon,cgcnn}.example.yaml`
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
  - `TailTrainConfig` standalone per un tail di classificazione (famiglia).
* - `tail_train_visualization.example.yaml`
  - `TailTrainConfig` standalone per un tail di visualizzazione.
* - `tail_train_hierarchical_supcon_best_combo.example.yaml`
  - esperti per famiglia (`hierarchical_supcon`) sulla combinazione di riferimento.
* - `tail_train_visualization_best_combo.example.yaml`
  - tail di visualizzazione sulla combinazione di riferimento.
* - `tail_train_*_encoder_256_128.example.yaml`
  - le stesse due, per l'encoder `[256, 128]`.
* - `tail_train_hierarchical_supcon_supcon_mace_*.example.yaml`
  - esperti per famiglia su una run `supcon_mace`.
* - `round19_sweep/*.example.yaml`
  - le nove varianti di larghezza/profondità del round 19.
```

## Riferimento completo

`examples/config_reference.example.yaml` elenca ogni campo con default/commenti — non pensato per essere lanciato così com'è, solo come promemoria dell'intera forma dello schema YAML.
