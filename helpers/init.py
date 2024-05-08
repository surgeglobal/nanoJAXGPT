"""
Author              : Sachith Gunasekara
Date Created        : 08/05/2023 11:31 AM
Purpose             : To add local implementations of the torch.nn.init written in jax for NumPy arrays
"""

import jax


def normal_(array: jax.Array, mean: float, std: float, key: jax.random.PRNGKey = jax.random.PRNGKey(0)) -> None:
    new_array = jax.random.normal(key, array.shape) * std + mean
    return new_array


def zeros_(array: jax.Array) -> None:
    new_array = jax.numpy.zeros(array.shape)
    return new_array