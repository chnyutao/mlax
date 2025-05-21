import equinox as eqx
import jax
import optax
import wandb
from tqdm.auto import tqdm

from mlax.datasets import make_mnist_dataset
from mlax.plot import plt

from .model import VAE, loss_fn

wandb.init(project='mlax', name='vae')

# config
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-4
N_PLOTS = 4
SEED = 42

# init dataset
dataset = make_mnist_dataset(train=True, flatten=True)

# init model
key, model_key = jax.random.split(jax.random.key(SEED))
model = VAE(model_key, input_size=784, latent_size=100, hidden_sizes=[200, 200])

# train loop
opt = optax.adam(LR)
opt_state = opt.init(eqx.filter(model, eqx.is_array))
for x, _ in tqdm(dataset.shuffle(SEED).batch(BATCH_SIZE).repeat(EPOCHS)):
    key, subkey = jax.random.split(key)
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, key=subkey)
    updates, opt_state = opt.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    wandb.log({'loss': loss})

# plot
x, _ = next(iter(dataset.batch(N_PLOTS)))
x_hat = jax.vmap(model)(x, key=jax.random.split(key, N_PLOTS))[0].clip(0, 1)
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
