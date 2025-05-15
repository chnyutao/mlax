# WANDB: https://wandb.ai/chnyutao/mlax/runs/2kal1jbc

from time import time_ns

import equinox as eqx
import jax
import jax.numpy as jnp
import wandb
from tqdm.auto import tqdm

from .datasets import make_function_dataset

wandb.init(project='mlax', name='lr')

BATCH_SIZE = 1
EPOCHS = 50
LR = 5e-3
SEED = 42


class LinearRegression(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, in_size: int, out_size: int):
        self.weight = jnp.ones((out_size, in_size))
        self.bias = jnp.zeros((out_size,))

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.weight @ x + self.bias


def loss_fn(model: LinearRegression, x: jax.Array, y: jax.Array) -> jax.Array:
    # mean squared error
    return jnp.mean((jax.vmap(model)(x) - y) ** 2, axis=-1)


if __name__ == '__main__':
    key = jax.random.key(SEED)
    dataset = make_function_dataset(key, lambda x: 3 * x - 1, eps=0.2)
    model = LinearRegression(in_size=1, out_size=1)
    for epoch in tqdm(range(EPOCHS)):
        for batch in dataset.shuffle(seed=time_ns()).batch(BATCH_SIZE):
            assert isinstance(batch, dict)
            x, y = batch.values()
            loss, grads = jax.value_and_grad(loss_fn)(model, x, y)
            model = jax.tree.map(lambda p, g: p - LR * g, model, grads)
            wandb.log({'loss': loss})
