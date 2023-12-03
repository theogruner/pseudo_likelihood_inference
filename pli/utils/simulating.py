from typing import Callable

import jax


def n_sims_per_param_wrapper(simulator: Callable, n_samples_per_param: int):
    """Wrapper for simulating multiple simulations per parameter."""
    def sim_wrapper(key, param):
        n_keys = jax.random.split(key, n_samples_per_param)
        return jax.vmap(simulator, (0, None), 0)(n_keys, param)

    return sim_wrapper


def n_params_wrapper(simulator: Callable):
    """Wrapper for simulating given multiple parameters."""
    def sim_wrapper(key, params):
        n_keys = jax.random.split(key, params.shape[0])
        return jax.vmap(simulator, (0, 0), 0)(n_keys, params)

    return sim_wrapper
