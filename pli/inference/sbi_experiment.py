from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PyTree

from pli.costs.statistical_distance import mse, wasserstein, memory_efficient_mmd
from pli.utils.jax_utils import batched_operations
from pli.utils.sampling import batched_sampling_tries, build_sampler
from pli.utils.simulating import n_sims_per_param_wrapper
from .abc.types import ABCTrainState
from .pseudo_likelihood_inference.types import PLITrainState
from .snpe.types import SNPETrainState


def run_and_eval_simulations(
    simulator,
    distance_fn,
    train_state,
    param_samples,
    ref_data,
):
    """
    Run simulations, calculate the distances and approximate log-partition function
    Typically this is the computationally heavy part of the experiment.
    :param train_state: The current state of the training
    :param param_samples: Parameter samples from the proposal distribution
    :param ref_data: The reference data. Currently, only supported for a single observation.
    :return: The updated training state and the evaluated distances and log-partition function
    """
    # 1. Simulate
    rng_key, next_rng_key = jax.random.split(train_state.rng_key)
    simulations = simulator(rng_key, param_samples)

    # 2. Calculate statistical distances
    distance = jax.vmap(distance_fn, (0, 0), 0)(ref_data, simulations)

    # 3. Calculate log-partition function
    train_state = train_state.replace(
        rng_key=next_rng_key,
        n_simulations=train_state.n_simulations
        + simulations.shape[0] * simulations.shape[1],
    )
    return train_state, {"distance": distance, "simulations": simulations}


def build_eval_fn(
    model,
    simulator,
    n_samples_per_param,
    sampling_batch_size,
    n_parallel_operations,
    conditioned_model: bool,
):
    sim_wrapper = n_sims_per_param_wrapper(simulator, n_samples_per_param)

    if n_samples_per_param == 1:
        wasserstein_fn = mse
        mmd_fn = mse
    else:
        mmd_fn = memory_efficient_mmd
        wasserstein_fn = wasserstein

    if conditioned_model:
        def log_prob_fn(params, data, model_params):
            return model.log_prob(
                params,
                data,
                model_params,
            )
    else:
        def log_prob_fn(params, data, model_params):
            return model.log_prob(
                params,
                model_params,
            )

    def eval_sbi_experiment(
        rng_key,
        train_state: Union[PLITrainState, ABCTrainState, SNPETrainState],
        data: jnp.ndarray,
        n_posterior_samples: int,
        param_support,
        n_simulation_samples: int = 1,
        ground_truth_params: Optional[jnp.ndarray] = None,
        posterior_samples: Optional[jnp.ndarray] = None,
        max_resampling_tries=1000,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, PyTree]:
        sample_key, ppc_key, posterior_choice_key = jax.random.split(rng_key, 3)

        n_eval_samples = n_posterior_samples

        sampler = build_sampler(
            model,
            train_state.model_params,
            data=data if conditioned_model else None,
        )

        # sampler = partial(posterior_sampler, data=data, model_params=train_state.model_params)

        # print("Start sampling")
        param_samples, tries = batched_sampling_tries(
            sample_key,
            n_eval_samples,
            sampler,
            param_support,
            batch_size=sampling_batch_size,
            max_tries=max_resampling_tries,
        )

        def eval_simulations(key, param):
            simulations = sim_wrapper(key, param)
            ppc_mmd = mmd_fn(data, simulations)
            ppc_wasserstein = wasserstein_fn(data, simulations)
            return ppc_mmd, ppc_wasserstein

        ppc_keys = jax.random.split(ppc_key, n_eval_samples)
        ppc_mmd, ppc_wasserstein = batched_operations(
            jax.vmap(eval_simulations, (0, 0), 0), n_parallel_operations, ppc_keys, param_samples
        )

        # We use the same rng key to match the distance.
        # To avoid memory issues, we restrict storing all simulations,
        # and instead use the same key on a fraction of all simulations.
        sims = jax.vmap(
            sim_wrapper, (0, 0), 0
        )(ppc_keys[:n_simulation_samples], param_samples[:n_simulation_samples])
        # test = mmd_fn(data, sims[0])

        logging_dict = {
            "tries": tries,
            "ppc wasserstein": ppc_wasserstein,
            "ppc mmd": ppc_mmd,
        }
        if ground_truth_params is not None:
            ground_truth_log_prob = log_prob_fn(
                ground_truth_params[jnp.newaxis, :],
                data[jnp.newaxis],
                train_state.model_params,
            )
            logging_dict["ground truth log prob"] = ground_truth_log_prob

        if posterior_samples is not None:
            # print("Start posterior evaluation")
            cropped_posterior_samples = jax.random.choice(
                posterior_choice_key,
                posterior_samples,
                shape=(n_posterior_samples,),
                replace=False,
            )

            posterior_log_prob = np.mean(
                log_prob_fn(
                    cropped_posterior_samples,
                    jnp.repeat(
                        data[jnp.newaxis], cropped_posterior_samples.shape[0], axis=0
                    ),
                    train_state.model_params,
                )
            )

            wasserstein_param_dist = wasserstein(cropped_posterior_samples, param_samples)
            mmd_param_dist = memory_efficient_mmd(cropped_posterior_samples, param_samples)

            logging_dict["posterior log prob"] = posterior_log_prob
            logging_dict["wasserstein param dist"] = wasserstein_param_dist
            logging_dict["mmd param dist"] = mmd_param_dist

        return (
            param_samples,
            sims,
            logging_dict,
        )
    return eval_sbi_experiment
