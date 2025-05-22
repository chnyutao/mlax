import sys

import equinox as eqx
import jax
import jax.random as jr
import wandb
from tqdm.auto import tqdm

from mlax.datasets import make_function_dataset
from mlax.plot import colors, plt

from .model import LinearRegression, loss_fn

wandb.init(project='mlax', name='lr')

# config
BATCH_SIZE = 1
EPOCHS = 50
LR = 5e-3
SEED = 42

# init dataset
key = jr.key(SEED)
dataset = make_function_dataset(key, lambda x: 3 * x - 1, eps=0.2)

# init model
model = LinearRegression(in_size=1, out_size=1)

# train loop
for x, y in tqdm(dataset.shuffle(SEED).batch(BATCH_SIZE).repeat(EPOCHS)):
    loss, grads = jax.value_and_grad(loss_fn)(model, x, y)
    model = eqx.apply_updates(model, jax.tree.map(lambda g: -LR * g, grads))
    wandb.log({'loss': loss})

# plot
x, y = next(iter(dataset.batch(sys.maxsize)))
plt.scatter(x, y, s=3, c=colors[0], label='Noisy Data')
plt.plot(x, jax.vmap(model)(x), c=colors[1], label='Fitted Line')
plt.legend()
# plt.show()
wandb.log({'plot': wandb.Image(plt)})
