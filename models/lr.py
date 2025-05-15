from typing import NamedTuple, Self

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import vmap


class LinearRegression(NamedTuple):
    weight: jax.Array
    bias: jax.Array

    @classmethod
    def init(cls, in_size: int, out_size: int) -> Self:
        weight = jnp.ones((out_size, in_size))
        bias = jnp.zeros((out_size,))
        return cls(weight, bias)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.weight @ x + self.bias


def loss_fn(model: LinearRegression, xs: jax.Array, ys: jax.Array) -> jax.Array:
    return jnp.mean((vmap(model)(xs) - ys) ** 2)


ETA = 5e-3

if __name__ == '__main__':
    key = jax.random.key(42)
    fn = vmap(lambda x: 3 * x + 1)
    key1, key2 = jax.random.split(key)
    xs = jax.random.normal(key1, (100, 1))
    eps = jax.random.normal(key2, (100, 1)) * 0.5
    ys = fn(xs) + eps
    model = LinearRegression.init(in_size=1, out_size=1)
    for epoch in range(1000):
        grads = jax.grad(loss_fn)(model, xs, ys)
        model = jax.tree.map(lambda param, grad: param - grad * ETA, model, grads)
    plt.scatter(xs, ys)
    plt.plot(xs, vmap(model)(xs), c='red')
    plt.show()
