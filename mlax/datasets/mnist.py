from datasets import Dataset, load_dataset


def make_mnist_dataset(train: bool = True) -> Dataset:
    """Loads MNIST dataset.

    Args:
        train (bool): Whether to load the training (True) or test (False) set. Defaults to True.

    Returns:
        Dataset: The MNIST dataset with the specified split.
    """
    split = 'train' if train else 'test'
    dataset = load_dataset('mnist', split=split)
    assert isinstance(dataset, Dataset)
    return dataset
