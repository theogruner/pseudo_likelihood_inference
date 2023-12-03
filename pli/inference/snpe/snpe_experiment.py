from typing import Tuple, Optional, Dict

import jax
import jax.numpy as jnp
import haiku as hk
import optax
from jaxtyping import Array, PyTree

from pli.inference.sbi_experiment import build_eval_fn
import pli.models.conditioned_density_estimators.conditioned_nsf as cnsf
from pli.models.utils import build_conditioned_density_estimator
from pli.utils.sampling import batched_sampling
from pli.utils.simulating import n_sims_per_param_wrapper, n_params_wrapper
from .neural_posterior_update import build_neural_posterior_update
from .types import SNPETrainState
from ..types import SBIExperiment


def build_snpe_experiment(
    rng_key: Array,
    model_name: str,
    param_support: Array,
    prior,
    simulator,
    sample_data,
    n_param_samples: int,
    n_samples_per_param: int,
    model_cfg: Optional[Dict] = None,
    sampling_batch_size: int = -1,
    n_parallel_operations: int = -1,
    max_resampling_tries: int = 1000,
    **_ignore,
) -> SBIExperiment:
    if sampling_batch_size == -1:
        sampling_batch_size = n_param_samples

    sim_wrapper = n_params_wrapper(n_sims_per_param_wrapper(simulator, n_samples_per_param))

    prior_key, sim_key = jax.random.split(rng_key)
    prior_samples = prior.sample(prior_key, (10_000,))
    simulations = sim_wrapper(sim_key, prior_samples)
    if model_cfg is None:
        model_cfg = {}
    model = build_conditioned_density_estimator(
        model_name,
        param_support,
        prior_samples,
        simulations,
        **model_cfg
    )

    update_method = model_cfg.pop("update_method", None)
    model_update = build_neural_posterior_update(
        update_method,
        prior,
        model,  # summary_statistic
        **model_cfg,
    )

    def init_fnc(rng_key) -> Tuple[SNPETrainState, PyTree]:
        model_init_key, rng_key = jax.random.split(rng_key)

        model_params = model.init(model_init_key, sample_data)
        optimizer = optax.adam(model_cfg["lr"])
        filter_non_trainable_params_fn = cnsf.filter_non_trainable_params
        trainable_params, _ = hk.data_structures.partition(
            filter_non_trainable_params_fn, model_params
        )
        opt_state = optimizer.init(trainable_params)

        train_state = SNPETrainState(
            rng_key=rng_key,
            model_params=model_params,
            optimizer=optimizer,
            model_opt_state=opt_state,
        )

        return train_state, {}

    def step_fnc(train_state, data):
        train_state = train_state.replace(episode=train_state.episode + 1)

        # summary_data = summary_statistic(data[jnp.newaxis])

        def first_round(train_state):
            rng_key, next_rng_key = jax.random.split(train_state.rng_key)

            param_samples = batched_sampling(
                rng_key, n_param_samples, prior.sample, param_support, sampling_batch_size, max_resampling_tries
            )

            # Simulate
            rng_key, next_rng_key = jax.random.split(next_rng_key)
            simulations = sim_wrapper(rng_key, param_samples)
            train_state = train_state.replace(
                rng_key=next_rng_key,
                n_simulations=train_state.n_simulations
                + simulations.shape[0] * simulations.shape[1],
            )

            train_state, tl = model_update(
                param_samples, simulations, train_state, True
            )

            return train_state, tl

        def multi_round(train_state):
            rng_key, next_rng_key = jax.random.split(train_state.rng_key)

            # Condition posterior on data
            def predictive_posterior_sampler(rng_key, sample_shape):
                return model.sample(
                    rng_key,
                    sample_shape,
                    jnp.repeat(data[jnp.newaxis], sample_shape[0], axis=0),
                    train_state.model_params,
                )

            param_samples = batched_sampling(
                rng_key,
                n_param_samples,
                predictive_posterior_sampler,
                param_support,
                sampling_batch_size,
                max_resampling_tries
            )

            # Simulate
            rng_key, next_rng_key = jax.random.split(next_rng_key)
            simulations = sim_wrapper(rng_key, param_samples)
            train_state = train_state.replace(
                rng_key=next_rng_key,
                n_simulations=train_state.n_simulations
                + simulations.shape[0] * simulations.shape[1],
            )
            train_state, tl = model_update(
                param_samples, simulations, train_state, False
            )

            return train_state, tl

        return jax.lax.cond(
            train_state.episode == 1, first_round, multi_round, train_state
        )

    eval_fnc = build_eval_fn(
        model,
        simulator,
        n_samples_per_param,
        sampling_batch_size,
        n_parallel_operations,
        conditioned_model=True
    )

    return SBIExperiment(init=init_fnc, step=step_fnc, evaluate=eval_fnc, model=model)
