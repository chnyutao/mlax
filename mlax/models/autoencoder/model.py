from collections.abc import Callable, Sequence
from itertools import pairwise

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from equinox import nn
from jaxtyping import Array, PRNGKeyArray


class AutoEncoder(eqx.Module):
    encoder: nn.Sequential
    decoder: nn.Sequential

    def __init__(
        self,
        layer_sizes: Sequence[int],
        activation: Callable[[Array], Array] = jax.nn.relu,
        *,
        key: PRNGKeyArray,
    ):
        """
        Initialize an auto-encoder with MLP encoder/decoder.

        Args:
            layer_sizes (`Sequence[int]`): Integers `(input_size, *hidden_sizes, latent_size)` that
                determine the encoder structure, and the reveresed decoder structure.
            activation (`Callable[[Array], Array]`, optional):
                Activation function. Defaults to `jax.nn.relu`.
            key (`PRNGKeyArray`): JAX random key.
        """
        keys = iter(jr.split(key, 2 * len(layer_sizes) - 2))
        # encoder
        layers = []
        for in_size, out_size in pairwise(layer_sizes):
            layers.append(nn.Linear(in_size, out_size, key=next(keys)))
            layers.append(nn.Lambda(activation))
        self.encoder = nn.Sequential(layers[:-1])  # remove last activation
        # decoder
        layers = []
        for in_size, out_size in pairwise(reversed(layer_sizes)):
            layers.append(nn.Linear(in_size, out_size, key=next(keys)))
            layers.append(nn.Lambda(activation))
        self.decoder = nn.Sequential(layers[:-1])  # remove last activation

    def __call__(self, x: Array) -> Array:
        """
        Auto-encode the input array.

        Args:
            x (`Array`): Input array of shape `(layer_sizes[0])`.

        Returns:
            Reconstructed array of shape `(layer_sizes[0])`.
        """
        return self.decoder(self.encoder(x))


@eqx.filter_jit
def loss_fn(model: AutoEncoder, x: Array) -> Array:
    """mean squared error"""
    x_hat = jax.vmap(model)(x)
    return jnp.mean(jnp.sum((x - x_hat) ** 2, axis=-1))
