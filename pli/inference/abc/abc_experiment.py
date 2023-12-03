from typing import Callable, Dict, Tuple
import math


import jax
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import PyTree

from pli.costs.statistical_distance import wasserstein, memory_efficient_mmd, mse
from pli.inference.sbi_experiment import build_eval_fn
from pli.models.particle_samplers.smc_sampler import build_smc_sampler
from pli.models.pseudo_likelihoods.utils import build_pseudo_likelihood
from pli.utils.jax_utils import batched_operations
from pli.utils.sampling import batched_sampling
from pli.utils.simulating import n_sims_per_param_wrapper
from .pmc_update import build_pmc_update
from .types import ABCTrainState
from ..types import SBIExperiment


def build_abc_experiment(
    rng_key: Array,
    model_name: str,
    pseudo_likelihood_name: str,
    param_support: Array,
    prior,
    simulator: Callable,
    sample_data: Array,
    n_param_samples: int,
    n_samples_per_param: int,
    model_cfg: Dict,
    pseudo_likelihood_cfg: Dict,
    statistical_distance: str = "mmd",
    max_resampling_tries: int = 10_000,
    sampling_batch_size: int = -1,
    n_parallel_operations: int = -1,
    **_ignore,
) -> SBIExperiment:
    if sampling_batch_size == -1:
        sampling_batch_size = n_param_samples

    sim_wrapper = n_sims_per_param_wrapper(simulator, n_samples_per_param)

    # Define distance
    if n_samples_per_param == 1:
        distance_measure = mse
    else:
        if statistical_distance == "mmd":
            distance_measure = memory_efficient_mmd
        elif statistical_distance == "wasserstein":
            distance_measure = wasserstein
        else:
            raise ValueError(f"Unknown distance measure: {statistical_distance}")

    if model_name == "PMCSampler":
        # In PMC, we keep a stack of the best particles based on the alpha-quantile
        # and only a fraction of the particles `n_param_samples' are exchanged in each iteration.
        n_init_param_samples = math.ceil(
            (
                (1 - model_cfg["alpha_quantile"])
                / model_cfg["alpha_quantile"]
                * n_param_samples
            )
        )
    else:
        raise NotImplementedError("Only the PMCSampler is currently implemented")

    # Build models
    model = build_smc_sampler(param_support=param_support, **model_cfg)

    # Same for both
    likelihood = build_pseudo_likelihood(
        pseudo_likelihood_name,
        simulator,
        n_samples_per_param=n_samples_per_param,
        distance_fn=distance_measure,
        **pseudo_likelihood_cfg,
    )

    # Build update methods for the model and the pseudo-likelihood
    model_update = build_pmc_update(prior, **model_cfg)

    if pseudo_likelihood_name == "UniformKernel":
        if pseudo_likelihood_cfg["update_method"] == "alpha_quantile":

            def pseudo_likelihood_update(
                distance, train_state: ABCTrainState
            ) -> ABCTrainState:
                epsilon = jnp.quantile(
                    jnp.concatenate([train_state.best_distance, distance], axis=0),
                    1 - pseudo_likelihood_cfg["alpha_quantile"],
                )
                return train_state.replace(
                    likelihood_params={"epsilon": epsilon},
                    # bandwidth=bandwidth(epsilon)
                )

        else:
            raise ValueError(
                f"The only supported update methods are `ess` or `alpha_quantile`, "
                f"but {pseudo_likelihood_cfg['update_method']} was given."
            )
    else:
        raise ValueError(
            f"Update methods currently only exist for the `UniformKernel`, but {pseudo_likelihood_name} was given."
        )

    def simulate_and_evaluate(keys, params, data):
        def _simulate_and_evaluate(key, param):
            sim = sim_wrapper(key, param)
            distance = distance_measure(data, sim)
            return distance

        return jax.vmap(_simulate_and_evaluate)(keys, params)

    def init_abc(key) -> Tuple[ABCTrainState, PyTree]:
        # Initialize likelihood and model parameters
        ll_init_key, model_init_key, key = jax.random.split(key, 3)
        likelihood_params = likelihood.init(ll_init_key)
        model_params = model.init(model_init_key, n_init_param_samples)
        train_state = ABCTrainState(
            rng_key=key,
            likelihood_params=likelihood_params,
            model_params=model_params,
            best_distance=jnp.zeros(n_init_param_samples, dtype=float),  # filler
            current_distance=jnp.zeros(n_param_samples, dtype=float),  # filler
            previous_likelihood=jnp.zeros(n_init_param_samples, dtype=float),  # filler
        )

        # Run initial step 0
        sampling_key, key = jax.random.split(train_state.rng_key)
        param_samples = batched_sampling(
            sampling_key,
            n_init_param_samples,
            prior.sample,
            param_support,
            sampling_batch_size,
            max_tries=max_resampling_tries,
        )

        next_key, key = jax.random.split(key)
        sim_keys = jax.random.split(next_key, param_samples.shape[0])
        dist = batched_operations(
            simulate_and_evaluate,
            n_parallel_operations,
            sim_keys,
            param_samples,
            data=sample_data[0],
        )
        log_partition = jnp.zeros_like(dist)

        lh = likelihood.pdf_from_distance(
            dist[:n_param_samples],
            log_partition[:n_param_samples],
            train_state.likelihood_params,
        )
        train_state = train_state.replace(
            rng_key=key,
            best_distance=dist,
            current_distance=jnp.zeros(n_param_samples, dtype=float),
            previous_likelihood=lh,
        )
        return train_state, {"lh": lh}

    def step_fnc(
        train_state: ABCTrainState,
        data: Array,
    ) -> Tuple[ABCTrainState, PyTree]:
        rng_key, next_rng_key = jax.random.split(train_state.rng_key)
        train_state = train_state.replace(
            rng_key=next_rng_key, episode=train_state.episode + 1
        )

        # Sample proposal parameters
        def first_round():
            samples = batched_sampling(
                rng_key,
                n_param_samples,
                prior.sample,
                param_support,
                batch_size=sampling_batch_size,
                max_tries=max_resampling_tries,
            )
            return samples

        def multi_round():
            def predictive_posterior_sampler(rng_key, sample_shape):
                return model.sample(
                    rng_key,
                    sample_shape,
                    train_state.model_params,
                )

            samples = batched_sampling(
                rng_key,
                n_param_samples,
                predictive_posterior_sampler,
                support=param_support,
                batch_size=sampling_batch_size,
                max_tries=max_resampling_tries,
            )
            return samples

        param_samples = jax.lax.cond(train_state.episode == 1, first_round, multi_round)

        # Update the bandwidth parameter
        # Only necessary for SMC-ABC
        sim_keys = jax.random.split(rng_key, param_samples.shape[0])
        distance = batched_operations(
            simulate_and_evaluate,
            n_parallel_operations,
            sim_keys,
            param_samples,
            data=sample_data[0],
        )
        train_state = train_state.replace(
            current_distance=distance,
            n_simulations=train_state.n_simulations
            + n_samples_per_param * n_param_samples,
        )
        log_partition = jnp.zeros_like(distance)
        train_state = pseudo_likelihood_update(distance, train_state)
        lh = likelihood.pdf_from_distance(
            distance, log_partition, train_state.likelihood_params
        )

        # Update model
        train_state, train_log = model_update(param_samples, lh, data, train_state)

        return train_state, train_log

    eval_fnc = build_eval_fn(
        model,
        simulator,
        n_samples_per_param,
        sampling_batch_size,
        n_parallel_operations,
        conditioned_model=False,
    )

    return SBIExperiment(init=init_abc, step=step_fnc, evaluate=eval_fnc, model=model)


def bisection(opt_fnc, low, high, threshold, max_iter=100):
    def bisec_(i, l, h):
        mp = (l + h) / 2.0
        # jax.debug.print("{}", opt_fnc(l) * opt_fnc(mp) < 0.0)
        return jax.lax.cond(
            opt_fnc(l) * opt_fnc(mp) < 0.0,
            lambda: (i + 1, l, mp),
            lambda: (i + 1, mp, h),
        )

    def cond_fn(i, l, h):
        return jnp.logical_and(((h - l) / 2.0) > threshold, i < max_iter)

    _, low, high = jax.lax.while_loop(
        lambda x: cond_fn(*x),
        lambda x: bisec_(*x),
        (0, low, high),
    )
    # jax.debug.print("epsilon: {}", low)
    return (low + high) / 2.0
