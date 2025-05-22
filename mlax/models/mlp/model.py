from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from equinox import nn
from jaxtyping import Array, PRNGKeyArray


class MLP(eqx.Module):
    layers: list
    activation: Callable[[Array], Array]

    def __init__(
        self,
        layer_sizes: Sequence[int],
        activation: Callable[[Array], Array] = jax.nn.relu,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize a Multi-Layer Perceptron (MLP) of `len(layer_sizes)-1` layers.

        Args:
            layer_sizes (`Sequence[int]`): Sizes of each layer in the network.
                Must contain at least two integers (input and output).
            activation (`Callable[[Array], Array]`, optional):
                Activation function. Defaults to `jax.nn.relu`.
            key (`PRNGKeyArray`): JAX random key.
        """
        assert len(layer_sizes) > 1
        keys = jr.split(key, len(layer_sizes) - 1)
        self.layers = [
            nn.Linear(layer_sizes[i], layer_sizes[i + 1], key=keys[i])
            for i in range(len(layer_sizes) - 1)
        ]
        self.activation = activation

    def __call__(self, x: Array) -> Array:
        """
        Forward input array through the layers of MLP.

        Args:
            x (`Array`): Input array of shape `(layer_sizes[0],)`.

        Returns:
            Output array of shape `(layer_sizes[-1],)`.
        """
        for layer in self.layers:
            x = self.activation(layer(x))
        return x


@eqx.filter_jit
def loss_fn(model: MLP, x: Array, y: Array) -> Array:
    """cross entropy loss"""
    p = jax.nn.log_softmax(jax.vmap(model)(x))
    return -jnp.sum(jnp.take_along_axis(p, y, axis=1))


@eqx.filter_jit
def accuracy(model: MLP, x: Array, y: Array) -> Array:
    """accuracy (%)"""
    p = jax.nn.log_softmax(jax.vmap(model)(x))
    return jnp.mean(p.argmax(axis=1, keepdims=True) == y)
