from collections.abc import Callable, Sequence
from itertools import pairwise

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import nn
from jaxtyping import PRNGKeyArray


class VAE(eqx.Module):
    encoder: nn.Sequential
    decoder: nn.Sequential

    def __init__(
        self,
        key: PRNGKeyArray,
        input_size: int,
        latent_size: int,
        hidden_sizes: Sequence[int] | None = None,
        activation: Callable[[jax.Array], jax.Array] = jax.nn.relu,
    ):
        """
        Initialize an auto-encoder with MLP encoder/decoder.

        Args:
            key (`PRNGKeyArray`): JAX random key.
            input_size (`int`): Input array size.
            latent_size (`int`): Latent Gaussian size.
            hidden_size (`Sequence[int]`): Sizes of each hidden layer of the encoder, also
                sizes of each hidden layer (reversed) of the decoder.
        """
        if hidden_sizes is None:
            hidden_sizes = []
        keys = iter(jax.random.split(key, 2 * len(hidden_sizes) + 2))
        # encoder
        layers = []
        for in_size, out_size in pairwise([input_size, *hidden_sizes, 2 * latent_size]):
            layers.append(nn.Linear(in_size, out_size, key=next(keys)))
            layers.append(nn.Lambda(activation))
        self.encoder = nn.Sequential(layers[:-1])  # remove last activation
        # decoder
        layers = []
        for in_size, out_size in pairwise([latent_size, *reversed(hidden_sizes), input_size]):
            layers.append(nn.Linear(in_size, out_size, key=next(keys)))
            layers.append(nn.Lambda(activation))
        self.decoder = nn.Sequential(layers[:-1])  # remove last activation

    def __call__(
        self, x: jax.Array, *, key: PRNGKeyArray
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """
        Auto-encode the input array.

        Args:
            x (`jax.Array`): Input array of shape `(layer_sizes[0])`.
            key (`PRNGKeyArray`): JAX random key.

        Returns:
            Reconstructed array of shape `(layer_sizes[0])`.
        """
        mu, log_var = jnp.split(self.encoder(x), 2)
        std = jnp.exp(0.5 * log_var)
        z = mu + std * jax.random.normal(key, std.shape)
        return self.decoder(z), mu, log_var


@eqx.filter_jit
def loss_fn(model: VAE, x: jax.Array, *, key: PRNGKeyArray) -> jax.Array:
    """negative evidence lower bound (ELBO)"""
    x_hat, mu, log_var = jax.vmap(model)(x, key=jax.random.split(key, x.shape[0]))
    reconst = jnp.sum((x - x_hat) ** 2, axis=-1)  # shape [B, 1]
    kl = 0.5 * jnp.sum(1 + log_var - mu**2 - jnp.exp(log_var), axis=-1)  # shape [B, 1]
    return jnp.mean(reconst - kl)
