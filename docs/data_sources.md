# Sorgenti dati & augmentation

`RunConfig.data_source` sceglie come viene costruito il database di strutture *prima* della featurizzazione: `"fetch"` (default) interroga Materials Project per strutture reali; `"pyxtal"` genera strutture sintetiche simmetriche con `dim_red.generate`. Esattamente uno tra `fetch`/`pyxtal` deve essere presente in YAML, coerentemente con `data_source` — la validazione è in `RunConfig.__post_init__` (vedi {doc}`runconfig`).

## `fetch:` — Materials Project reale

```{eval-rst}
.. autoclass:: dim_red.pipeline.config.FetchConfig

```

## `pyxtal:` — strutture sintetiche generate

```{eval-rst}
.. autoclass:: dim_red.pipeline.config.PyxtalConfig

```

Uno spacegroup i cui moltiplicatori di Wyckoff non riescono ad accomodare il conteggio richiesto genera solo un avviso di log durante la generazione, non un errore fatale.

## `augmentation:` — jitter e vacanze, opzionale

Applicata dopo la costruzione del dataset (`fetch` o `pyxtal`, indifferentemente) e prima della featurizzazione. `None` (default di `RunConfig.augmentation`) la disabilita del tutto.

```{eval-rst}
.. autoclass:: dim_red.pipeline.config.AugmentationConfig

```

```{note}
Le impostazioni di augmentation entrano nella chiave della cache del dataset — una config che cambia solo `augmentation` non riutilizza una cache costruita senza (o con impostazioni diverse).
```
