from collections.abc import Callable

import grain
import jax
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray


def make_function_dataset(
    key: PRNGKeyArray,
    fn: Callable[[Array], Array],
    eps: float = 0.5,
    range_: tuple[float, float] = (0, 1),
    shape: tuple[int, int] = (100, 1),
) -> grain.MapDataset:
    """
    Initialize a synthetic dataset for a given function.

    Args:
        key (`PRNGKeyArray`): JAX random key.
        fn (`Callable[[Array], Array]`): A function that maps array inputs to outputs.
        eps (`float`, optional): Noise scale factor. Defaults to 0.5.
        range\\_ (`tuple[float, float]`, optional): Range for sampling inputs. Defaults to (0, 1).
        shape (`tuple[int, int]`, optional): Shape of the inputs (n_samples, n_features).
            Defaults to (100, 1).

    Returns:
        A dataset containing input and outupt pairs (x, y).
    """
    input_key, noise_key = jr.split(key)
    x = jr.uniform(input_key, shape, minval=range_[0], maxval=range_[1])
    y = jax.vmap(fn)(x)
    y = y + eps * jr.normal(noise_key, y.shape)
    return grain.MapDataset.source(list(zip(x, y)))
