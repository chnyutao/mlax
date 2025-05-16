import equinox as eqx
import jax
import jax.numpy as jnp


class LinearRegression(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, in_size: int, out_size: int):
        self.weight = jnp.ones((out_size, in_size))
        self.bias = jnp.zeros((out_size,))

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.weight @ x + self.bias


def loss_fn(model: LinearRegression, x: jax.Array, y: jax.Array) -> jax.Array:
    """mean squared error"""
    return jnp.mean((jax.vmap(model)(x) - y) ** 2)
