# FullStack (supcon / supcon_mace)

`supcon` e `supcon_mace` si allenano **solo** tramite `FullStack` (`dim_red.pipeline.full_stack`). Il vecchio `RunConfig`/`run_single` è ormai solo `cgcnn` (`model: cgcnn`, vedi {doc}`model_kinds`).

## Cos'è

Uno stack "family" più fino a 7 esperti indipendenti, tutti con la **stessa architettura** (`dim_red.supcon.stack.SingleStack`): encoder + projection tail, poi classification tail e visualization tail. Cambiano solo etichetta, dati e valori di config.

- `family`: label = sistema cristallino.
- un esperto per sistema cristallino (`triclinic`, `monoclinic`, `orthorhombic`, `tetragonal`, `trigonal`, `hexagonal`, `cubic`): label = spacegroup. Esistono solo gli esperti elencati in config; si può allenare anche un solo stack, senza `family`.

Ogni stack ha due sotto-fasi: **body** (encoder + projection insieme, loss SupCon) e **heads** (encoder congelato; classification e visualization tail sull'embedding; la projection è esclusa). Body e heads sono comandi separati: si possono allenare più set di heads (`--heads-name`) sullo stesso body.

Ogni stack costruisce un proprio dataset, split (grouped per `material_id`), standardizzazione e seed (`run_seed + indice` in `STACK_ORDER`, salvo `seed` esplicito nel blocco).

## Config

Esempi: `configs/full_stack.example.yaml`, `configs/full_stack_best_combo.example.yaml`, `configs/full_stack_sweep.example.yaml`, `configs/train_heads.example.yaml`.

- Chiavi top-level: `name`, `seed`, `model_kind` (`supcon` | `supcon_mace`), `output_dir`, `family`, `experts`. Il parser è stretto: chiavi sconosciute sono un errore.
- Blocco di uno stack: `data` (`pyxtal`, `augmentation`, `soap`, `mace`), `encoder`, `projection`, `contrastive`, `classifier`, `viz`, `train` (incluso `early_stopping`), `batching`, `seed`, `min_train_rows`.
- `experts.defaults` è unito (deep merge) sotto il blocco di ogni esperto; ogni esperto può sovrascrivere singoli campi. Uno stack viene risolto in uno `StackSpec` completo prima di costruire qualunque dataset.
- Solo `pyxtal` come sorgente dati (Materials Project non è supportato). Un esperto genera solo il proprio sistema cristallino; `spacegroups` espliciti devono appartenergli.
- `model_kind: supcon` richiede `data.soap.element_agnostic: true` (compressione SOAP `mu2`: le feature dipendono solo dalla geometria, non dalle specie chimiche presenti nel dataset); `supcon_mace` richiede `data.mace.checkpoint_path`.
- Un esperto con troppe poche righe di training o meno di 2 spacegroup è un errore esplicito (nessun fallback).

## Comandi

```{list-table}
:header-rows: 1
:widths: 30 70

* - Comando
  - Cosa fa
* - `dimred-run <config>`
  - Dispatch sulle chiavi della config (`model_kind`/`family`/`experts` = FullStack, altrimenti cgcnn). Per FullStack: body di tutti gli stack, poi heads `default`.
* - `dimred-rerun <run_dir>`
  - Rilancia una run da `config.yaml` risolto, in una nuova directory.
* - `dimred-sweep <config>`
  - Solo FullStack. `base` + `grid` con percorsi puntati (`family.encoder.latent_dim`, `experts.defaults.train.epochs`); scrive `sweep.yaml` nella cartella della sweep.
* - `dimred-train-heads <config> <run_dir> --heads-name NAME [--stacks a,b]`
  - Nuovo set di heads su body già allenati. Della config si leggono, per ogni stack elencato, solo classifier/viz/seed.
* - `dimred-compare <sweep_dir>`
  - Una suite di grafici per stack. `--heads-name` se ci sono più set di heads.
* - `dimred-benchmark <run_o_sweep>... --output <csv>`
  - Una riga per stack. `--heads-name` se ci sono più set di heads.
* - `dimred-apply <structures.extxyz> <run_dir>`
  - Per run FullStack: CSV di predizioni + npz delle viz in `<run_dir>/applied`.
* - `dimred-train-tail`
  - Solo cgcnn, vedi {doc}`tails`.
```

## Su disco

```text
<run_dir>/
  config.yaml                  # FullStackConfig risolta
  stacks/<family|cubic|...>/
    config.yaml                # StackSpec risolto
    dataset.extxyz  classes.yaml  embeddings.npz
    body/                      # encoder_params, projection_params, stack.yaml, loss_history.csv
    heads/<nome>/              # parametri classifier/viz, heads.yaml,
                               # classifier_loss_history.csv, viz_loss_history.csv,
                               # predictions.npz, viz_embeddings.npz,
                               # viz_plot.png (solo se viz_dim == 2)
```

Le directory `.<x>.tmp` sono scritture atomiche interrotte: i lettori (`run_layout`) le ignorano. Scrivere il body in una cartella `stacks/<nome>/` già esistente è un errore, non una sovrascrittura. I run con il vecchio layout (pre-FullStack) **non** sono leggibili.

## Predire

```python
from dim_red.pipeline.full_stack import FullStack

prediction = FullStack.open(run_dir).predict(structures)  # device="cpu" di default
```

Lo stack `family` assegna il sistema cristallino, poi ogni struttura è instradata all'esperto di quel sistema. Se l'esperto non è allenato i campi `expert`/`spacegroup`/`spacegroup_proba` sono `None`. `device="cpu"` funziona sempre, anche per run allenati su GPU. `predict_stack(name, structures)` usa un solo stack senza instradamento. Le feature delle nuove strutture sono calcolate da `pipeline.featurize` (SOAP `mu2` o MACE) e standardizzate con le statistiche salvate nello stack.

## Quirk noti

- `dimred-rerun` di un run che vive dentro una directory di sweep lo ricrea dentro quella directory (l'`output_dir` registrato) e con la cache dei dataset dello sweep; per rilanciarlo altrove copiare la config e cambiare `output_dir`.
- Le chiavi di history `*_family_supcon` compaiono anche negli esperti, dove contrastano spacegroup; i grafici le rietichettano.
- I dataset di `family` e degli esperti non condividono strutture: la valutazione end-to-end della catena famiglia → esperto richiede un set held-out (`examples/evaluate_holdout_pyxtal.py`).
- `tests/golden/make_golden.py` serve solo come provenienza dei riferimenti golden (girava sul codice pre-FullStack, commit `a693fa2`).

```{eval-rst}
.. autoclass:: dim_red.pipeline.full_stack.FullStack
   :members: create, open, fit_body, predict, predict_stack
```
