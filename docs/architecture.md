# Architettura encoder/decoder condivisa

Blocco YAML `vae:`, usato da {bdg-primary}`vae` {bdg-info}`autoencoder` {bdg-warning}`supcon` (che ignora `decoder_hidden_dim`/`mirror`, non avendo decoder) e in parte da {bdg-success}`cgcnn` (solo `latent_dim`). Mantenuto con questo nome per compatibilità con sweep esistenti anche quando il `model_kind` non è `"vae"`.

```{eval-rst}
.. autoclass:: dim_red.pipeline.config.VAEArchConfig

```
