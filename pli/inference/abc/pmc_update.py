from typing import Tuple, Dict

import jax
import jax.numpy as jnp

from pli.inference.abc.types import ABCTrainState
from pli.models.particle_samplers.markov_kernels.kernels import build_markov_kernel
from pli.utils.sampling import ess


def build_pmc_update(prior, markov_kernel: str, **_ignore):
    markov_kernel = build_markov_kernel(markov_kernel)

    def update_method(params, likelihood, ref_data, train_state):
        return population_monte_carlo(
            params, likelihood, ref_data, train_state, markov_kernel, prior
        )

    return update_method


def population_monte_carlo(
    params, likelihood, ref_data, train_state: ABCTrainState, markov_kernel, prior
) -> Tuple[ABCTrainState, Dict]:
    """Sampling procedure of Adaptive Population Monte-Carlo [1].

    [1] Lenormand, M., Jabot, F., & Deffuant, G. (2013).
        Adaptive approximate Bayesian computation for complex models.
    """
    n_particles = train_state.model_params["particles"].shape[0]
    particles = train_state.model_params["particles"]
    weights = train_state.model_params["weights"]

    # Evaluate probability of the samples parameters
    # We initialize the numerator with ones as we assume a uniform prior
    # If this changes in the future, we need to change the numerator accordingly (see [1]).
    numerator = jnp.exp(prior.log_prob(params))
    denominator = jnp.sum(
        weights
        * markov_kernel.forward(
            params,
            particles,
            train_state.model_params["kernel_params"],
        ),
        axis=1,
    )
    proposal_weights = numerator / denominator

    # Concatenate the newly sampled parameters with the previous collection
    proposal_distance = jnp.concatenate(
        [train_state.best_distance, train_state.current_distance]
    )
    proposal_particles = jnp.concatenate(
        [train_state.model_params["particles"], params]
    )
    proposal_weights = jnp.concatenate([weights, proposal_weights])

    accept_idx = jnp.where(
        proposal_distance <= train_state.likelihood_params["epsilon"],
        jnp.arange(proposal_distance.shape[0]),
        proposal_distance.shape[0],
    )
    accepted_idx = jnp.sort(accept_idx)

    padded_proposal_particles = jnp.concatenate(
        [proposal_particles, jnp.zeros((1, proposal_particles.shape[-1]))]
    )
    padded_proposal_weights = jnp.concatenate([proposal_weights, jnp.array([0.0])])
    padded_proposal_distance = jnp.concatenate([proposal_distance, jnp.array([0.0])])
    updated_particles = padded_proposal_particles[accepted_idx][:n_particles]
    updated_weights = padded_proposal_weights[accepted_idx][:n_particles] / jnp.sum(
        padded_proposal_weights[accepted_idx][:n_particles]
    )
    rng_key, next_rng_key = jax.random.split(train_state.rng_key)
    updated_kernel_params = markov_kernel.update(
        rng_key, updated_particles, updated_weights
    )
    train_state = train_state.replace(
        rng_key=next_rng_key,
        model_params=dict(
            particles=updated_particles,
            weights=updated_weights,
            kernel_params=updated_kernel_params,
        ),
        best_distance=padded_proposal_distance[accepted_idx][:n_particles],
        model_update_steps=train_state.model_update_steps + 1,
    )
    return train_state, {
        "ess": ess(weights),
        "pmc steps": train_state.model_update_steps,
    }
