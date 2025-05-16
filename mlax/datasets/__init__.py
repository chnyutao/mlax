from datasets.utils import disable_progress_bar

from .function import make_function_dataset
from .mnist import make_mnist_dataset

__all__ = ['make_function_dataset', 'make_mnist_dataset']

disable_progress_bar()
