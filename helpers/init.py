"""
Author              : Sachith Gunasekara
Date Created        : 08/05/2023 11:31 AM
Purpose             : To add local implementations of the torch.nn.init written in jax for NumPy arrays
"""

import jax


def normal_(array: jax.Array, mean: float, std: float, key: jax.random.PRNGKey = jax.random.PRNGKey(0)) -> None:
    array = jax.random.normal(key, array.shape) * std + mean


def zeros_(array: jax.Array) -> None:
    array = jax.numpy.zeros(array.shape)