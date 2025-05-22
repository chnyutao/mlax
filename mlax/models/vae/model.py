from collections.abc import Callable, Sequence
from itertools import pairwise

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from equinox import nn
from jaxtyping import Array, PRNGKeyArray


class VAE(eqx.Module):
    encoder: nn.Sequential
    decoder: nn.Sequential

    def __init__(
        self,
        input_size: int,
        latent_size: int,
        hidden_sizes: Sequence[int] | None = None,
        activation: Callable[[Array], Array] = jax.nn.relu,
        *,
        key: PRNGKeyArray,
    ):
        """
        Initialize an auto-encoder with MLP encoder/decoder.

        Args:
            input_size (`int`): Input array size.
            latent_size (`int`): Latent Gaussian size.
            hidden_size (`Sequence[int]`, optional): Sizes of each hidden layer of the encoder, also
                sizes of each hidden layer (reversed) of the decoder.
            activation (`Callable[[Array], Array]`, optional):
                Activation function. Defaults to `jax.nn.relu`.
            key (`PRNGKeyArray`): JAX random key.
        """
        if hidden_sizes is None:
            hidden_sizes = []
        keys = iter(jr.split(key, 2 * len(hidden_sizes) + 2))
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

    def __call__(self, x: Array, *, key: PRNGKeyArray) -> tuple[Array, Array, Array]:
        """
        Auto-encode the input array.

        Args:
            x (`Array`): Input array of shape `(layer_sizes[0])`.
            key (`PRNGKeyArray`): JAX random key.

        Returns:
            Reconstructed array of shape `(layer_sizes[0])`.
        """
        mu, log_var = jnp.split(self.encoder(x), 2)
        std = jnp.exp(0.5 * log_var)
        z = mu + std * jr.normal(key, std.shape)
        return self.decoder(z), mu, log_var


@eqx.filter_jit
def loss_fn(model: VAE, x: Array, *, key: PRNGKeyArray) -> Array:
    """negative evidence lower bound (ELBO)"""
    x_hat, mu, log_var = jax.vmap(model)(x, key=jr.split(key, x.shape[0]))
    reconst = jnp.sum((x - x_hat) ** 2, axis=-1)  # shape [B, 1]
    kl = 0.5 * jnp.sum(1 + log_var - mu**2 - jnp.exp(log_var), axis=-1)  # shape [B, 1]
    return jnp.mean(reconst - kl)
