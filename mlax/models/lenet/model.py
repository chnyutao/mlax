from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from equinox import nn
from jax import lax
from jaxtyping import Array, PRNGKeyArray


def as_sliding_windows(x: Array, kernel_size: int, stride: int) -> Array:
    """
    Extracts sliding window patches from a 2D input array.

    Args:
        x (`Array`): Input array of shape (h, w).
        kernel_size (`int`): Size of the (square) sliding window.
        stride (`int`): Step size between consecutive windows.

    Returns:
        Output array of shape (h', w', kernel_size, kernel_size), where\\
        h' = ⌈ (h - kernel_size + 1) / stride ⌉,\\
        w' = ⌈ (w - kernel_size + 1) / stride ⌉.
    """
    rows = jnp.arange(0, x.shape[0] - kernel_size + 1, stride)
    cols = jnp.arange(0, x.shape[1] - kernel_size + 1, stride)
    # cartesian product (rows x cols)
    indices = jnp.stack([ax.ravel() for ax in jnp.meshgrid(rows, cols)], axis=-1)
    # generate sliding windows x[r:r+K, c:c+K]
    _, windows = lax.scan(
        lambda _, index: (_, lax.dynamic_slice(x, index, (kernel_size,) * 2)),
        None,
        indices.reshape(-1, 2),
    )
    return windows.reshape(*rows.shape, *cols.shape, kernel_size, kernel_size)


class Conv2d(eqx.Module):
    weight: Array
    bias: Array
    kernel_size: int = eqx.field(static=True)
    padding: int | Sequence[int] | Sequence[tuple[int, int]] = eqx.field(static=True)
    stride: int = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int | Sequence[int] | Sequence[tuple[int, int]] = 0,
        stride: int = 1,
        *,
        key: PRNGKeyArray,
    ):
        """
        Initializes a 2D convolutional layer.

        Args:
            in_channels (`int`): Number of input channels.
            out_channels (`int`): Number of output channels.
            kernel_size (`int`): Size of the (square) convolutional kernel.
            padding (`int` | `Sequence[int]` | `Sequence[tuple[int, int]]`, optional):
                Input padding sizes (refer to `jnp.pad`). Defaults to 0.
            stride (`int`, optional): Stride of the convolutional kernel. Defaults to 1.
            key (`PRNGKeyArray`): JAX random key.
        """
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        # init weight
        lim = 1 / jnp.sqrt(in_channels * kernel_size * kernel_size)
        wkey, bkey = jr.split(key)
        self.weight = jr.uniform(
            wkey,
            (out_channels, in_channels, kernel_size, kernel_size),
            minval=-lim,
            maxval=lim,
        )
        self.bias = jr.uniform(bkey, (out_channels, 1, 1), minval=-lim, maxval=lim)

    def __call__(self, x: Array) -> Array:
        """
        Apply convolutional kernels to input array patches by sliding windows.

        Args:
            x: Input array of shape (c, h, w).

        Returns:
            Output array of shape (c, h', w') where\\
            h' = ⌈ (pad_h - kernel_size + 1) / stride ⌉,\\
            w' = ⌈ (pad_w - kernel_size + 1) / stride ⌉.
        """
        windows = jax.vmap(as_sliding_windows, in_axes=(0, None, None))(
            # not pad the leading channel dimension
            jax.vmap(jnp.pad, in_axes=(0, None))(x, self.padding),
            self.kernel_size,
            self.stride,
        )
        return jnp.einsum('oijk,ihwjk->ohw', self.weight, windows) + self.bias


class Pooling2d(eqx.Module):
    kernel_size: int = eqx.field(static=True)
    padding: int | Sequence[int] | Sequence[tuple[int, int]] = eqx.field(static=True)
    pooling_fn: Callable[..., Array]
    stride: int = eqx.field(static=True)

    def __init__(
        self,
        kernel_size: int,
        padding: int | Sequence[int] | Sequence[tuple[int, int]] = 0,
        pooling_fn: Callable[..., Array] = jnp.mean,
        stride: int = 1,
    ):
        """
        Initialize a 2D pooling layer.

        Args:
            kernel_size (`int`): Size of the (square) pooling window.
            padding (`int` | `Sequence[int]` | `Sequence[tuple[int, int]]`, optional):
                Input padding sizes (refer to `jnp.pad`). Defaults to 0.
            pooling_fn (`Callable[..., Array]`, optional): Pooling function, e.g. `jnp.mean`
                for average pooling and `jnp.max` for max pooling. Defaults to `jnp.mean`.
            stride (`int`, optional): Stride of the pooling window. Defaults to 1.
        """
        self.kernel_size = kernel_size
        self.padding = padding
        self.pooling_fn = pooling_fn
        self.stride = stride

    def __call__(self, x: Array) -> Array:
        """
        Aggregate input array patches by sliding windows.

        Args:
            x: Input array of shape (c, h, w).

        Returns:
            Output array of shape (c, h', w') where\\
            h' = ⌈ (pad_h - kernel_size + 1) / stride ⌉,\\
            w' = ⌈ (pad_w - kernel_size + 1) / stride ⌉.
        """
        windows = jax.vmap(as_sliding_windows, in_axes=(0, None, None))(
            # not pad the leading channel dimension
            jax.vmap(jnp.pad, in_axes=(0, None))(x, self.padding),
            self.kernel_size,
            self.stride,
        )
        return self.pooling_fn(windows, axis=(-2, -1))


class LeNet(eqx.Module):
    layers: list

    def __init__(
        self,
        activation: Callable[[Array], Array] = jax.nn.relu,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize a LeNet[^1], the classic convolutional neural network (CNN) for MNIST.

        [^1]: LeCun, Yann, et al. "Gradient-based learning applied to document recognition." \
            *Proceedings of the IEEE* 86.11 (1998): 2278-2324.

        Args:
            activation (`Callable[[Array], Array]`): Activation function
            key (`PRNGKeyArray`): JAX random key.
        """
        keys = jr.split(key, 5)
        self.layers = [
            Conv2d(1, 6, kernel_size=5, padding=2, key=keys[0]),
            activation,
            Pooling2d(kernel_size=2, stride=2),
            Conv2d(6, 16, kernel_size=5, key=keys[1]),
            activation,
            Pooling2d(kernel_size=2, stride=2),
            jnp.ravel,
            nn.Linear(16 * 5 * 5, 120, key=keys[2]),
            activation,
            nn.Linear(120, 84, key=keys[3]),
            activation,
            nn.Linear(84, 10, key=keys[4]),
        ]

    def __call__(self, x: Array) -> Array:
        """
        Forward input array through the layers of MLP.

        Args:
            x (`Array`): Input array of shape (28, 28).

        Returns:
            Output array of shape (10,).
        """
        for layer in self.layers:
            x = layer(x)
        return x


@eqx.filter_jit
def loss_fn(model: LeNet, x: Array, y: Array) -> Array:
    """cross entropy loss"""
    p = jax.nn.log_softmax(jax.vmap(model)(x))
    return -jnp.sum(jnp.take_along_axis(p, y, axis=1))


@eqx.filter_jit
def accuracy(model: LeNet, x: Array, y: Array) -> Array:
    """accuracy (%)"""
    p = jax.nn.log_softmax(jax.vmap(model)(x))
    return jnp.mean(p.argmax(axis=1, keepdims=True) == y)
