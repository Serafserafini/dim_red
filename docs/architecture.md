# Architettura dell'encoder

Blocco YAML `encoder:`, usato da {bdg-warning}`supcon` e {bdg-secondary}`supcon_mace` per dimensionare l'MLP del corpo, e da {bdg-success}`cgcnn` solo per `latent_dim` (la sua rete a grafo è dimensionata da `graph:`).

```{note}
Questo blocco si chiamava `vae:` quando esistevano i modelli VAE e autoencoder (rimossi). Il vecchio nome `vae:` viene ancora letto — e unito sotto `encoder:` — così le config e i `config.yaml` salvati dalle run precedenti continuano a caricarsi; le chiavi `decoder_hidden_dim` e `mirror` sono ignorate.
```

```{eval-rst}
.. autoclass:: dim_red.pipeline.config.EncoderConfig

```
