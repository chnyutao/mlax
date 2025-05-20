from collections.abc import Callable, Sequence
from itertools import pairwise

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import nn
from jaxtyping import PRNGKeyArray


class AutoEncoder(eqx.Module):
    encoder: nn.Sequential
    decoder: nn.Sequential

    def __init__(
        self,
        key: PRNGKeyArray,
        layer_sizes: Sequence[int],
        activation: Callable[[jax.Array], jax.Array] = jax.nn.relu,
    ):
        """
        Initialize an auto-encoder with MLP encoder/decoder.

        Args:
            key (`PRNGKeyArray`): JAX random key.
            layer_sizes (`Sequence[int]`): Integers `(input_size, *hidden_sizes, latent_size)` that
                determine the encoder structure, and the reveresed decoder structure.
        """
        keys = iter(jax.random.split(key, 2 * len(layer_sizes) - 2))
        layers = []
        for in_size, out_size in pairwise(layer_sizes):
            layers.append(nn.Linear(in_size, out_size, key=next(keys)))
            layers.append(nn.Lambda(activation))
        self.encoder = nn.Sequential(layers[:-1])  # remove last activation
        layers = []
        for in_size, out_size in pairwise(reversed(layer_sizes)):
            layers.append(nn.Linear(in_size, out_size, key=next(keys)))
            layers.append(nn.Lambda(activation))
        self.decoder = nn.Sequential(layers[:-1])  # remove last activation

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Auto-encode the input array.

        Args:
            x (`jax.Array`): Input array of shape `(layer_sizes[0])`.

        Returns:
            Reconstructed array of shape `(layer_sizes[0])`.
        """
        return self.decoder(self.encoder(x))


@eqx.filter_jit
def loss_fn(model: AutoEncoder, x: jax.Array) -> jax.Array:
    """mean squared error"""
    x_hat = jax.vmap(model)(x)
    return jnp.mean(jnp.sum((x - x_hat) ** 2, axis=-1))
