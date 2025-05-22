import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array


class LinearRegression(eqx.Module):
    weight: Array
    bias: Array

    def __init__(self, in_size: int, out_size: int):
        """
        Initialize a linear regression model.

        Args:
            in_size (`int`): Input dimension.
            out_size (`int`): Output dimension.
        """
        self.weight = jnp.ones((out_size, in_size))
        self.bias = jnp.zeros((out_size,))

    def __call__(self, x: Array) -> Array:
        """Linearly transform the input data.

        Args:
            x (`Array`): Input array of shape (in_size,).

        Returns:
            Output array of shape (out_size,).
        """
        return self.weight @ x + self.bias


@jax.jit
def loss_fn(model: LinearRegression, x: Array, y: Array) -> Array:
    """mean squared error"""
    return jnp.mean((jax.vmap(model)(x) - y) ** 2)
