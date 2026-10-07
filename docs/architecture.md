# Architettura dell'encoder

Blocco YAML `encoder:`, usato da {bdg-success}`cgcnn` solo per `latent_dim` (la sua rete a grafo è dimensionata da `graph:`). Gli stack di {bdg-warning}`supcon` / {bdg-secondary}`supcon_mace` hanno un blocco `encoder:` per ogni stack (`encoder_hidden_dim`, `latent_dim`), vedi {doc}`fullstack`.

```{note}
Questo blocco si chiamava `vae:` quando esistevano i modelli VAE e autoencoder (rimossi). Il vecchio nome `vae:` viene ancora letto — e unito sotto `encoder:` — nelle config cgcnn; le chiavi `decoder_hidden_dim` e `mirror` sono ignorate.
```

```{eval-rst}
.. autoclass:: dim_red.pipeline.config.EncoderConfig

```
