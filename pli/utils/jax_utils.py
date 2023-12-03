from functools import partial
from typing import Callable
import numpy as np
import jax
import jax.numpy as jnp


def tree_equal(tree1, tree2):
    return jax.tree_util.tree_all(
            jax.tree_util.tree_map(lambda x, y: jnp.allclose(x, y), tree1, tree2)
        )


@partial(jax.jit, static_argnames=("fnc", "batch_size"))
def batched_operations(fnc: Callable, batch_size, *batch_args, **kwargs):
    """Distribute operations to smaller batch-sizes to prevent memory overflow.
    Note, that this function itself is not jittable. All functions that are passed to this function must be jittable.

    :param fnc: Function to be batched
    :param args: Arguments of the function
    :param batch_size: Batch size for sampling. Please ensure that n_samples % batch_size == 0
    :return: Results of the function
    """

    def batched_fnc(x, args):
        return None, fnc(*args, **kwargs)

    n = batch_args[0].shape[0]

    if batch_size == -1 or batch_size > n:
        batch_size = n

    n_batches = np.ceil(n / batch_size).astype(int)
    remaining_batches = (batch_size - (n % batch_size)) % batch_size

    batch_args = jax.tree_util.tree_map(
        lambda x: jnp.concatenate([x, jnp.broadcast_to(x[-1][jnp.newaxis], (remaining_batches, *x.shape[1:]))]),
        batch_args
    )

    batched_args = jax.tree_util.tree_map(lambda x: x.reshape((n_batches, batch_size, *x.shape[1:])), batch_args)

    _, out = jax.lax.scan(batched_fnc, None, batched_args)
    return jax.tree_util.tree_map(lambda x: x.reshape((n + remaining_batches, *x.shape[2:])).at[:n].get(), out)