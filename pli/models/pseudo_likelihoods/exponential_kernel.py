from typing import Dict

import jax
import jax.numpy as jnp

from .types import PseudoLikelihood
from ...utils.jax_utils import batched_operations
from ...utils.simulating import n_sims_per_param_wrapper


def build_exponential_kernel(
    simulator,
    distance_fn,
    n_samples_per_param: int = 1,
    beta: float = 1e-3,
    n_parallel_operations=None,
    **ignore
):
    sim_wrapper = n_sims_per_param_wrapper(simulator, n_samples_per_param)

    if n_parallel_operations is None:
        n_parallel_operations = -1

    def init_exponential_kernel(rng_key) -> Dict:
        # Initialize the log-partition function
        return dict(beta=beta)

    def sample_fn(key, params):
        sample_keys = jax.random.split(key, params.shape[0])
        return batched_operations(jax.vmap(sim_wrapper, (0, 0), 0), n_parallel_operations, sample_keys, params)

    def log_prob_fn(rng_key, observations, parameters, state_dict: Dict):
        """
        :param params: Simulation parameters
        :param state_dict:
        :return:
        """
        def fn(keys, parameters, observations, state_dict):
            return jax.vmap(_log_prob_fn, (0, None, 0, None), 0)(keys, observations, parameters, state_dict)

        keys = jax.random.split(rng_key, parameters.shape[0])
        return batched_operations(
            fn, n_parallel_operations, keys, parameters, observations=observations, state_dict=state_dict
        )

    def _log_prob_fn(key, observations, parameter, state_dict: Dict):
        sim_key, partition_key = jax.random.split(key, 2)
        simulations = sim_wrapper(sim_key, parameter)
        distance = distance_fn(observations, simulations)
        return - distance / state_dict["beta"]

    def pdf_fn(rng_key, observations, parameters, state_dict: Dict):
        log_prob = log_prob_fn(rng_key, observations, parameters, state_dict)
        return jnp.exp(log_prob)  # , info

    def log_prob_from_distance(distance, log_partition, state_dict: Dict):
        return - distance / state_dict["beta"] - log_partition

    def pdf_from_distance(distance, log_partition, state_dict: Dict):
        return jnp.exp(log_prob_from_distance(distance, log_partition, state_dict))

    def bandwidth(state_dict: Dict):
        return state_dict["beta"].squeeze()

    return PseudoLikelihood(
        init=init_exponential_kernel,
        sample=sample_fn,
        log_prob=log_prob_fn,
        pdf=pdf_fn,
        log_prob_from_distance=log_prob_from_distance,
        pdf_from_distance=pdf_from_distance,
        bandwidth=bandwidth
    )
