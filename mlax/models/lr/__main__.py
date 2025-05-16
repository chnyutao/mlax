from time import time_ns

import jax
import matplotlib.pyplot as plt
import wandb
from tqdm.auto import tqdm

from mlax.datasets import make_function_dataset

from .model import LinearRegression, loss_fn

wandb.init(project='mlax', name='lr')

# config
BATCH_SIZE = 1
EPOCHS = 50
LR = 5e-3
SEED = 42

# init dataset
key = jax.random.key(SEED)
dataset = make_function_dataset(key, lambda x: 3 * x - 1, eps=0.2)

# init model
model = LinearRegression(in_size=1, out_size=1)

# train loop
for epoch in tqdm(range(EPOCHS)):
    for batch in dataset.shuffle(time_ns()).batch(BATCH_SIZE):
        assert isinstance(batch, dict)
        x, y = batch.values()
        loss, grads = jax.value_and_grad(loss_fn)(model, x, y)
        model = jax.tree.map(lambda p, g: p - LR * g, model, grads)
        wandb.log({'loss': loss})

# plot
x, y = dataset['x'], dataset['y']
plt.scatter(x, y)
plt.plot(x, jax.vmap(model)(x), c='r')  # type: ignore
# plt.show()
wandb.log({'plot': wandb.Image(plt)})
