import equinox as eqx
import jax
import jax.numpy as jnp
import wandb
from tqdm.auto import tqdm

from mlax.datasets import make_mnist_dataset

from .model import MLP, accuracy, loss_fn

wandb.init(project='mlax', name='mlp')

# config
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-4
SEED = 42

# init dataset
train_set = make_mnist_dataset(train=True, flatten=True, onehot=False)
test_set = make_mnist_dataset(train=False, flatten=True, onehot=False)

# init model
key = jax.random.key(SEED)
model = MLP(key, layer_sizes=[28 * 28, 128, 10])

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
