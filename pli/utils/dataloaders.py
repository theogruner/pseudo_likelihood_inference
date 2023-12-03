import logging
from typing import Union

import numpy as np
import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class Dataset:
    data: jnp.ndarray

    def update(self, new_data: jnp.ndarray):
        new_data = jnp.atleast_2d(new_data)
        new_size = new_data.shape[0]
        return self.replace(data=new_data, size=new_size)

    @property
    def size(self):
        return self.data.shape[0]

    @classmethod
    def create(cls, data: Union[np.ndarray, jnp.ndarray]):
        return cls(jnp.atleast_2d(data))


def build_dataloader(dataset: Dataset, batch_size):
    size = dataset.data.shape[0]

    if size < batch_size:
        logging.warning(
            f"The number of data entries is smaller than the batch size. "
            f"The batch size is set to {size} "
        )

        def batched_data_fn(rng_key: jnp.ndarray):
            perm_idx = permutation_indices(rng_key, dataset.data.shape[0])
            return shuffle_data(dataset.data, perm_idx)[jnp.newaxis, ...]

    else:

        def batched_data_fn(rng_key: jnp.ndarray):
            return build_batched_data(rng_key, dataset.data, batch_size)

    return batched_data_fn


def permutation_indices(rng_key, size):
    return jax.random.choice(
        rng_key,
        jnp.arange(size),
        shape=(size,),
        replace=False,
    )


def shuffle_data(data, perm_idx):
    return data.at[perm_idx].get()


def batched_data(data, batch_size):
    n_data_points = data.shape[0]
    num_batches, batch_size = n_data_points // batch_size, batch_size
    return jnp.asarray(
        jnp.split(data.at[: batch_size * num_batches].get(), num_batches)
    )


def build_batched_data(
    rng_key: jnp.ndarray, data: jnp.ndarray, batch_size: int
) -> jnp.ndarray:
    n_data_points = data.shape[0]
    perm_idx = permutation_indices(rng_key, n_data_points)
    shuffled_data = shuffle_data(data, perm_idx)

    return batched_data(shuffled_data, batch_size)


def batched_data_from_perm_idx(data, perm_idx, batch_size: int):
    shuffled_data = shuffle_data(data, perm_idx)
    return batched_data(shuffled_data, batch_size)
