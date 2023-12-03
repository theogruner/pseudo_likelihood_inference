from typing import Callable, Tuple, Optional, Dict

import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize
from jax.nn import log_softmax
from jax.scipy.special import logsumexp
from jaxtyping import Array
from jaxtyping import PyTree
import haiku as hk

from pli.costs.statistical_distance import wasserstein, mse, memory_efficient_mmd
from pli.utils.sampling import sample_within_support, batched_sampling
from pli.models.utils import build_density_estimator
from pli.models.pseudo_likelihoods.utils import build_pseudo_likelihood
from pli.inference.sbi_experiment import build_eval_fn
from pli.utils.jax_utils import batched_operations
from .model_update.base import build_model_update, build_optimizer
from .types import PLITrainState
from ..types import SBIExperiment


def build_pli_experiment(
    rng_key: Array,
    model_name: str,
    pseudo_likelihood_name: str,
    param_support: Array,
    prior,
    simulator: Callable,
    sample_data: Array,
    n_param_samples: int,
    n_samples_per_param: int,
    epsilon: float,
    model_cfg: Dict,
    pseudo_likelihood_cfg: Dict,
    statistical_distance: str = "mmd",
    max_resampling_tries: int = 1000,
    nu: float = 1.,
    sampling_batch_size: int = -1,
    n_parallel_operations: int = -1,
) -> SBIExperiment:
    """
    Initializes the pseudo-likelihood inference experiment.

    :param rng_key: A jax prng-key
    :param model_name: The name of the density estimator.
    Choose between "NSF", "Gaussian", "GMM", "PMCSampler", "SMCSampler"
    :param pseudo_likelihood_name:
    :param param_support: The support region in which the posterior samples lie.
    :param prior: A distribution that captures the prior believe over the parameters.
    :param simulator: A function mapping parameters to observations.
    The simulator emulates a stochastic model.
    :param sample_data: A sample observation to initialize the experiment.
    :param n_param_samples: Number of parameter samples to train on
    :param epsilon: The trust-region threshold.
    :param n_samples_per_param:
    :param model_cfg:
    :param pseudo_likelihood_cfg:
    :param statistical_distance: The statistical distance to use. For `n_samples_per_param == 1`,
    the choice defaults to the MSE.
    :param max_resampling_tries: Number of resampling tries to sample within the prior region.
    :param nu: A temperature parameter that regularizes the posterior on the prior.
    :param sampling_batch_size: The batch size to use for sampling from the density estimator.
    If the model is large or `n_param_samples` is large,
    setting the batch size works against overflow.
    :param n_parallel_operations: The number of parallel operations to use for simulating
    and calculating the statistical distance.
    :return: All that is required to run the experiment
    """
    if sampling_batch_size == -1:
        sampling_batch_size = n_param_samples

    # Define distance
    if n_samples_per_param == 1:
        distance_measure = mse
    else:
        if statistical_distance == "mmd":
            distance_measure = memory_efficient_mmd
        elif statistical_distance == "wasserstein":
            distance_measure = wasserstein
        else:
            raise ValueError("Statistical distance is not supported.")

    # Build models
    prior_sample = prior.sample(rng_key, (10_000,))
    model = build_density_estimator(
        name=model_name,
        param_support=param_support,
        prior_sample=prior_sample,
        **model_cfg,
    )

    likelihood = build_pseudo_likelihood(
        pseudo_likelihood_name,
        simulator,
        distance_fn=distance_measure,
        n_samples_per_param=n_samples_per_param,
        n_parallel_operations=n_parallel_operations,
        **pseudo_likelihood_cfg
    )

    # Build update methods for the model and the pseudo-likelihood
    model_update = build_model_update(
        model_name, model, **model_cfg
    )

    def estimate_weights(train_state: PLITrainState, params, ll_estimates):
        """
        Estimates the likelihood kernel based on the distance between the reference data
        and the simulated data
        :param train_state:
        :param params: Distance with Dim[batch_size]
        :param ll_estimates:
        :returns:
        """
        eta = jnp.exp(train_state.log_eta)
        prior_log_prob = prior.log_prob(params)
        prior_log_prob = jnp.where(prior_log_prob <= -1e10, -1e10, prior_log_prob)

        model_log_prob = batched_operations(
            model.log_prob, sampling_batch_size, params, model_params=train_state.model_params
        )

        log_importance_weights = train_state.nu / (train_state.nu + eta) * (
                prior_log_prob - model_log_prob
        )
        return jnp.exp(log_softmax(
            log_importance_weights + ll_estimates / (train_state.nu + eta)
        ) + jnp.log(ll_estimates.shape[0]))

    def dual_optimization(train_state: PLITrainState, params, ll_estimate, is_prior_proposal: bool):
        """Updating the temperature parameter of the tempered pseudo-likelihood.
        :param train_state:
        :param params: Parameters drawn from the density estimator
        :param ll_estimate: Estimate of the log-likelihoods for the params
        :param is_prior_proposal: Whether the parameter samples were drawn
        from the prior distribution. This is only true for the first training iteration.
        """
        log_eta = train_state.log_eta
        prior_log_prob = prior.log_prob(params)
        prior_log_prob = jnp.where(prior_log_prob <= -1e10, -1e10, prior_log_prob)

        model_log_prob = jax.lax.cond(
            is_prior_proposal,
            lambda: prior_log_prob,
            lambda: batched_operations(
                model.log_prob, sampling_batch_size, params, model_params=train_state.model_params
            ),
        )

        def dual_fnc(log_eta):
            eta = jnp.exp(log_eta).squeeze()
            importance_weights = jnp.exp(
                train_state.nu / (train_state.nu + eta) * (prior_log_prob - model_log_prob)
            )
            objective = eta * train_state.epsilon + (eta + train_state.nu) * (
                    logsumexp(ll_estimate / (train_state.nu + eta), axis=0, b=importance_weights)
                    - jnp.log(ll_estimate.shape[0])
            )
            return objective

        opt_dict = {"fun": dual_fnc, "x0": log_eta, "method": 'BFGS', "tol": 1e-08}
        res = minimize(**opt_dict)
        return train_state.replace(
            log_eta=res.x,
            # bandwidth=bandwidth(res.x)
        )

    def init_pli(rng_key) -> Tuple[PLITrainState, PyTree]:
        # Initialize likelihood and model parameters
        sampling_key, rng_key1, rng_key2, next_rng_key = jax.random.split(rng_key, 4)

        param_samples = batched_sampling(
            sampling_key,
            n_param_samples,
            prior.sample,
            param_support,
            sampling_batch_size,
            max_tries=max_resampling_tries,
        )

        likelihood_params = likelihood.init(rng_key1)
        model_params = model.init(rng_key2)
        # sum(x.size for x in jax.tree_leaves(model_params)) / 1e6
        opt_state = None
        optimizer, filter_non_trainable_params_fn = build_optimizer(model_name, **model_cfg)
        if optimizer is not None:
            trainable_params, _ = hk.data_structures.partition(
                filter_non_trainable_params_fn, model_params
            )
            opt_state = optimizer.init(trainable_params)
        train_state = PLITrainState(
            rng_key=next_rng_key,
            likelihood_params=likelihood_params,
            model_params=model_params,
            optimizer=optimizer,
            model_opt_state=opt_state,
            log_eta=jnp.log(jnp.array([(1e5 / likelihood_params["beta"]) - nu])),
            nu=nu,
            epsilon=epsilon
        )

        rng_key, next_rng_key = jax.random.split(train_state.rng_key)

        pseudo_lh = likelihood.pdf(
            rng_key,
            sample_data[0],
            param_samples,
            train_state.likelihood_params
        )
        train_state = train_state.replace(rng_key=next_rng_key)
        return train_state, {"lh": pseudo_lh}

    def step_fnc(
        train_state: PLITrainState,
        data: Array,
    ) -> Tuple[PLITrainState, PyTree]:
        rng_key, next_rng_key = jax.random.split(train_state.rng_key)
        train_state = train_state.replace(
            rng_key=next_rng_key, episode=train_state.episode + 1
        )

        # Sample proposal parameters
        def first_round():
            samples = sample_within_support(
                rng_key,
                n_param_samples,
                prior.sample,
                param_support,
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

        # If in first round, sample from the prior
        param_samples = jax.lax.cond(
            train_state.episode == 1, first_round, multi_round
        )

        # Approximate the pseudo-likelihood
        next_rng_key, rng_key = jax.random.split(train_state.rng_key)
        pseudo_llh = likelihood.log_prob(
            next_rng_key, data, param_samples, train_state.likelihood_params
        )
        train_state = train_state.replace(
            n_simulations=train_state.n_simulations + n_param_samples * n_samples_per_param
        )

        # Estimate the temperature parameter and calculate the weights
        train_state = dual_optimization(
            train_state, param_samples, pseudo_llh, train_state.episode == 1
        )
        weights = estimate_weights(train_state, param_samples, pseudo_llh)

        # Update model
        train_state, train_log = model_update(param_samples, weights, train_state)
        return train_state, train_log

    sbi_eval_fnc = build_eval_fn(
        model,
        simulator,
        n_samples_per_param,
        sampling_batch_size,
        n_parallel_operations,
        conditioned_model=False
    )

    def eval_fnc(
        rng_key,
        train_state: PLITrainState,
        data: Array,
        n_posterior_samples: int,
        param_support,
        n_simulation_samples: int = 1,
        ground_truth_params: Optional[Array] = None,
        posterior_samples: Optional[Array] = None,
        max_resampling_tries: int = 1000,
    ) -> Tuple[Array, Array, PyTree]:
        param_samples, sims, metrics = sbi_eval_fnc(
            rng_key,
            train_state,
            data,
            n_posterior_samples,
            param_support,
            n_simulation_samples,
            ground_truth_params,
            posterior_samples,
            max_resampling_tries
        )
        bandwidth = train_state.log_eta
        metrics["bandwidth"] = bandwidth
        return param_samples, sims, metrics

    return SBIExperiment(init=init_pli, step=step_fnc, evaluate=eval_fnc, model=model)
