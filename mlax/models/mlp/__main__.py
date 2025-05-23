import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import wandb
from tqdm.auto import tqdm

from mlax.datasets import make_mnist_dataset
from mlax.plot import plt

from .model import MLP, accuracy, loss_fn

wandb.init(project='mlax', name='mlp')

# config
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-4
N_PLOTS = 4
SEED = 42

# init dataset
train_set = make_mnist_dataset(train=True, flatten=True, onehot=False)
test_set = make_mnist_dataset(train=False, flatten=True, onehot=False)

# init model
key = jr.key(SEED)
model = MLP(layer_sizes=[28 * 28, 128, 10], key=key)

# train loop
for _ in tqdm(range(EPOCHS)):
    # training
    for x, y in train_set.shuffle(SEED).batch(BATCH_SIZE):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
        model = eqx.apply_updates(model, jax.tree.map(lambda g: -LR * g, grads))
        wandb.log({'loss': loss})
    # evaluation
    accuracies = []
    for x, y in test_set.shuffle(SEED).batch(BATCH_SIZE, drop_remainder=True):
        accuracies.append(accuracy(model, x, y))
    wandb.log({'accuracy': jnp.mean(jnp.array(accuracies))})

# plot
x, _ = next(iter(test_set.batch(N_PLOTS)))
y = jax.vmap(model)(x).argmax(axis=1)
for idx in range(N_PLOTS):
    plt.subplot(1, N_PLOTS, idx + 1)
    plt.imshow(x[idx].reshape(28, 28), cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title(f'Label={y[idx]}', fontsize=8)
# plt.show()
wandb.log({'plot': wandb.Image(plt)})
