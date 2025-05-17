import grain
import jax
import jax.numpy as jnp
from torchvision.datasets import MNIST


def make_mnist_dataset(train: bool = True, flatten: bool = True) -> grain.MapDataset:
    """Load the MNIST database of handwritten digits.

    Args:
        train (`bool`, optional): Whether to load the train or test set. Defaults to `True`.
        flatten (`bool`, optional): Whether to flatten images. Defaults to `True`.

    Returns:
        The MNIST database of handwritten digits.
    """
    dataset = MNIST(root='./data', train=train, download=True)
    dataset = grain.MapDataset.source(dataset)  # type: ignore
    if flatten:
        dataset = dataset.map(
            lambda data: (
                jnp.array(data[0]).ravel() / 255.0,
                jax.nn.one_hot(data[1], num_classes=10),
            )
        )
    else:
        dataset = dataset.map(
            lambda data: (
                jnp.array(data[0]) / 255.0,
                jax.nn.one_hot(data[1], num_classes=10),
            )  # fmt: skip
        )
    return dataset
