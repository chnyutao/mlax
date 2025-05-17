from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import nn
from jaxtyping import PRNGKeyArray


class MLP(eqx.Module):
    layers: list
    activation: Callable[[jax.Array], jax.Array]

    def __init__(
        self,
        key: PRNGKeyArray,
        layer_sizes: Sequence[int],
        activation: Callable[[jax.Array], jax.Array] = jax.nn.relu,
    ):
        """Initialize a Multi-Layer Perceptron (MLP) of `len(layer_sizes)-1` layers.

        Args:
            key (`PRNGKeyArray`): JAX random key.
            layer_sizes (`Sequence[int]`): Sizes of each layer in the network.
                Must contain at least two integers (input and output).
            activation (`Callable[[jax.Array], jax.Array]`, optional):
                The activation function. Defaults to `jax.nn.relu`.
        """
        assert len(layer_sizes) > 1
        keys = jax.random.split(key, len(layer_sizes) - 1)
        self.layers = [
            nn.Linear(layer_sizes[i], layer_sizes[i + 1], key=keys[i])
            for i in range(len(layer_sizes) - 1)
        ]
        self.activation = activation

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward input array through the layers of MLP.

        Args:
            x (`jax.Array`): Input array of shape `(layer_sizes[0],)`.

        Returns:
            Output array of shape `(layer_sizes[-1],)`.
        """
        for layer in self.layers:
            x = jax.nn.relu(layer(x))
        return x


@eqx.filter_jit
def loss_fn(model: MLP, x: jax.Array, y: jax.Array) -> jax.Array:
    """cross entropy loss"""
    p = jax.nn.log_softmax(jax.vmap(model)(x), axis=-1)
    return -jnp.sum(p * y)


@eqx.filter_jit
def accuracy(model: MLP, x: jax.Array, y: jax.Array) -> jax.Array:
    """accuracy (%)"""
    p = jax.nn.log_softmax(jax.vmap(model)(x), axis=-1)
    return jnp.mean(p.argmax(axis=-1) == y.argmax(axis=-1))
