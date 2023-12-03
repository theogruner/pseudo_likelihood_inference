from typing import Dict

import jax
import jax.numpy as jnp

from .types import PseudoLikelihood
from ...utils.jax_utils import batched_operations
from ...utils.simulating import n_sims_per_param_wrapper


def build_uniform_kernel(
        simulator,
        distance_fn,
        n_samples_per_param: int = 1,
        n_parallel_operations=None,
        use_log_partition=False,
        **_ignore
):
    if n_parallel_operations is None:
        n_parallel_operations = -1

    if use_log_partition:
        raise NotImplementedError("Log partition not implemented for uniform kernel")

    sim_wrapper = n_sims_per_param_wrapper(simulator, n_samples_per_param)

    def init_uniform_kernel(rng_key):
        epsilon = jnp.array(jnp.inf)
        return dict(epsilon=epsilon)

    def sample_fn(key, params):
        sample_keys = jax.random.split(key, params.shape[0])
        return batched_operations(jax.vmap(sim_wrapper, (0, 0), 0), n_parallel_operations, sample_keys, params)

    def pdf_fn(rng_key, observations, parameters, state_dict: Dict):

        def fn(keys, parameters, observations, state_dict):
            return jax.vmap(_pdf_fn, (0, None, 0, None), 0)(keys, observations, parameters, state_dict)
        keys = jax.random.split(rng_key, parameters.shape[0])
        return batched_operations(
            fn, n_parallel_operations, keys, parameters, observations=observations, state_dict=state_dict
        )

    def _pdf_fn(key, observations, parameter, state_dict: Dict):
        simulations = sim_wrapper(key, parameter)
        distance = distance_fn(observations, simulations)
        return jnp.asarray(distance <= state_dict["epsilon"], dtype="float32")

    def log_prob_fn(rng_key, observations, parameters, state_dict: Dict):
        pdf, info = pdf_fn(rng_key, observations, parameters, state_dict)
        return jnp.log(pdf)

    def pdf_from_distance(distance, log_partition, state_dict: Dict):
        return jnp.asarray(
            distance <= state_dict["epsilon"], dtype="float32"
        )

    def log_prob_from_distance(distance, log_partition, state_dict: Dict):
        return pdf_from_distance(distance, log_partition, state_dict)

    def bandwidth(state_dict: Dict):
        return state_dict["epsilon"].squeeze()

    return PseudoLikelihood(
        init=init_uniform_kernel,
        sample=sample_fn,
        log_prob=log_prob_fn,
        pdf=pdf_fn,
        log_prob_from_distance=log_prob_from_distance,
        pdf_from_distance=pdf_from_distance,
        bandwidth=bandwidth,
    )
