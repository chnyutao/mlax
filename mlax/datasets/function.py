from collections.abc import Callable

import jax
from datasets import Dataset
from jaxtyping import Array, PRNGKeyArray


def make_function_dataset(
    key: PRNGKeyArray,
    fn: Callable[[Array], Array],
    shape: tuple[int, int] = (100, 1),
    range_: tuple[float, float] = (0, 1),
    eps: float = 1,
) -> Dataset:
    """
    Initializes a synthetic dataset for a given function.

    Args:
        key (PRNGKeyArray): JAX random key.
        fn (Callable[[Array], Array]): A function that maps array inputs to outputs.
        shape (tuple[int, int], optional):
            Shape of the inputs (n_samples, n_features). Defaults to (100, 1).
        range\\_ (tuple[float, float], optional): Range for sampling inputs. Defaults to (0, 1).
        eps (float, optional): Noise epsilon. Defaults to 1.

    Returns:
        Dataset: A dataset object containing input 'x' and output 'y' arrays.
    """
    input_key, noise_key = jax.random.split(key)
    x = jax.random.uniform(input_key, shape, minval=range_[0], maxval=range_[1])
    y = jax.vmap(fn)(x)
    y = y + eps * jax.random.normal(noise_key, y.shape)
    return Dataset.from_dict({'x': x, 'y': y}).with_format('jax')
