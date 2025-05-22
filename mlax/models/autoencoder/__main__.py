import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import wandb
from tqdm.auto import tqdm

from mlax.datasets import make_mnist_dataset
from mlax.plot import plt

from .model import AutoEncoder, loss_fn

wandb.init(project='mlax', name='autoencoder')

# config
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-4
N_PLOTS = 4
SEED = 42

# init dataset
dataset = make_mnist_dataset(train=True, flatten=True)

# init model
key = jr.key(SEED)
model = AutoEncoder(layer_sizes=[28 * 28, 256, 64], key=key)

# train loop
for x, _ in tqdm(dataset.shuffle(SEED).batch(BATCH_SIZE).repeat(EPOCHS)):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x)
    model = eqx.apply_updates(model, jax.tree.map(lambda g: -LR * g, grads))
    wandb.log({'loss': loss})

# plot
x, _ = next(iter(dataset.batch(N_PLOTS)))
x_hat = jnp.clip(jax.vmap(model)(x), 0, 1)
plt.figure(figsize=(4, 2))
for row, (title, x) in enumerate({'Original': x, 'Reconstruction': x_hat}.items()):
    for col in range(N_PLOTS):
        plt.subplot(2, N_PLOTS, row * N_PLOTS + col + 1)
        plt.imshow(x[col].reshape(28, 28), cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.ylabel(f'{title if col == 0 else ""}', fontsize=8)
# plt.show()
wandb.log({'plot': wandb.Image(plt)})
