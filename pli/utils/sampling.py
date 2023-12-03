import warnings
from functools import partial
from typing import Callable, Tuple, Union
import numpy as np

import jax
import jax.numpy as jnp
import jax.experimental.host_callback as hcb
from jaxtyping import Array
from jaxtyping import PyTree

from pli.models.density_estimators.types import DensityEstimator


def build_sampler(
    model: DensityEstimator, model_params, data=None
) -> Callable[[jax.random.PRNGKey, Tuple, Array, PyTree], Array]:
    """Build a sampler for the posterior distribution.
    """
    if data is not None:
        def posterior_sampler(rng_key, sample_shape):
            # This should be an input as it can deviate between conditioned and unconditioned models
            return model.sample(
                rng_key,
                sample_shape,
                jnp.repeat(
                    data[jnp.newaxis], sample_shape[0], axis=0
                ),
                model_params,
            )
    else:
        def posterior_sampler(rng_key, sample_shape):
            return model.sample(
                rng_key,
                sample_shape,
                model_params,
            )
    return posterior_sampler


@partial(jax.jit, static_argnames=("sampler", "n_samples", "batch_size"))
def batched_sampling_tries(rng_key, n_samples, sampler, support, batch_size=-1, max_tries=1000):
    """Distribute sampling to smaller batch-sizes to prevent memory overflow.

    :param rng_key: A jax prng-key
    :param n_samples: Number of samples drawn from the sampler
    :param sampler: Sampling function to draw from
    :param batch_size: Batch size for sampling. Please ensure that n_samples % batch_size == 0
    :param support: Pre-defined ranges in which to sample
    :param max_tries: Maximum number of consecutive sampling tries
    :return: Samples
    """
    if batch_size == -1 or batch_size > n_samples:
        batch_size = n_samples

    n_batches = np.ceil(n_samples / batch_size).astype(int)

    def sample(rng_key, x):
        sample_key, next_rng_key = jax.random.split(rng_key)
        batched_samples, tries = sample_within_support_tries(
            sample_key, batch_size, sampler, support, max_tries
        )
        return next_rng_key, (batched_samples, tries)

    _, data = jax.lax.scan(sample, rng_key, None, length=n_batches)
    samples, tries = data
    return jnp.concatenate(samples, axis=0).at[:n_samples].get(), jnp.sum(tries)


def batched_sampling(rng_key, n_samples, sampler, support, batch_size=-1, max_tries=1000):
    samples, tries = batched_sampling_tries(rng_key, n_samples, sampler, support, batch_size, max_tries)
    return samples


def sample_within_support_tries(
    rng_key: Array,
    n_samples: int,
    sampler: Callable,
    support: Array,
    max_tries: int = 1000,
):
    """Sample from the distribution, but within the predefined support

    :param rng_key: A jax prng-key
    :param n_samples: Number of samples drawn from the sampler
    :param sampler: Sampling function to draw from
    :param support: Pre-defined ranges in which to sample
    :param max_tries: Maximum number of consecutive sampling tries
    :return: Samples
    """
    init_samples = jnp.zeros((n_samples, support.shape[1]))
    init_acceptance_mask = jnp.zeros(n_samples, dtype=jnp.bool_)

    def resample_fnc(resampling_data):
        """
        Resampling function that fills the remaining empty spots of the drawn_samples array.
        """
        next_rng_key, it, acceptance_mask, samples = resampling_data
        rng_key, next_rng_key = jax.random.split(next_rng_key)

        # Draw new proposal samples
        proposal_samples = sampler(rng_key, (n_samples,))

        # Evaluate whether new samples are within support
        below_prior_supremum = jnp.all(
            jax.lax.lt(proposal_samples, jnp.tile(support[1], (n_samples, 1))), axis=1
        )
        above_prior_infimum = jnp.all(
            jax.lax.gt(proposal_samples, jnp.tile(support[0], (n_samples, 1))), axis=1
        )
        acc_mask = jnp.logical_and(below_prior_supremum, above_prior_infimum)
        new_param_mask = jnp.logical_and(acc_mask, jnp.logical_not(acceptance_mask))
        acceptance_mask = jnp.logical_or(acceptance_mask, new_param_mask)

        # Add accepted samples to the already drawn ones
        samples = jnp.where(
            jnp.tile(new_param_mask[:, jnp.newaxis], (1, support.shape[-1])),
            proposal_samples,
            samples,
        )
        return next_rng_key, it + 1, acceptance_mask, samples

    def cond_fnc(resampling_data):
        """
        Check for successful sampling or if `max_tries` is exceeded.
        """
        _, it, acceptance_mask, _ = resampling_data
        return jnp.logical_and(jnp.sum(acceptance_mask) != n_samples, it != max_tries)

    _, tries, acceptance_mask, drawn_samples = jax.lax.while_loop(
        cond_fnc, resample_fnc, (rng_key, 0, init_acceptance_mask, init_samples)
    )
    hcb.id_tap(callback_warn, (max_tries, tries, acceptance_mask))
    return drawn_samples, tries


def sample_within_support(
    rng_key: Array,
    n_samples: int,
    sampler: Callable,
    support: Union[Array, np.ndarray],
    max_tries: int = 1000,
):
    drawn_samples, _ = sample_within_support_tries(rng_key, n_samples, sampler, support, max_tries)
    return drawn_samples


def callback_warn(x, transform):
    max_tries, tries, acceptance_mask = x
    if tries == max_tries:
        warnings.warn(
            f"Exceeding the maximum number of resampling tries {max_tries} "
            f"and only found {acceptance_mask.shape[0] - np.sum(acceptance_mask) } "
            f"out of {acceptance_mask.shape[0]} samples!\n"
            f"Consider interrupting the experiment and changing the configurations.", UserWarning
        )


def systematic_sampling(rng_key, weights):
    """Remove zero probability weights by systematic resampling [1]

    [1] https://github.com/pierrejacob/winference/blob/master/R/systematic_resampling.R

    :param rng_key: A jax prng-key
    :param weights: Weights of an empirical distribution. Presumably weights from particles
    :return: Ancestors with non-zero probability
    """

    # Normalize weights if they do not sum to 1
    weights = weights / weights.sum()
    n = weights.shape[0]

    def find_ancestors(data, x=None):
        """Systematic sampling to find the ancestors"""
        k, j, ancestors, csw, u = data
        csw, j = jax.lax.while_loop(
            lambda x: x[0] < u,
            lambda x: (x[0] + weights.at[x[1] + 1].get(), x[1] + 1),
            (csw, j),
        )
        return (k + 1, j, ancestors.at[k].set(j), csw, u + 1.0 / n), None

    u_init = jax.random.uniform(rng_key) / n

    # Ancestors fill the zero valued weights with
    # the closest ancestors that have non-zero probability
    ancestors = jnp.zeros(n, dtype=np.int)
    csw_init = weights[0]
    data, _ = jax.lax.scan(
        find_ancestors, (0, 0, ancestors, csw_init, u_init), xs=None, length=n
    )
    _, _, ancestors, _, _ = data
    return ancestors


def ess(particle_weights):
    """Effective Sample Size,
    a measure for the variance of particles.

    :param particle_weights: Weights of an empirical distribution. Presumably weights from particles
    """
    return 1 / jnp.sum((particle_weights / jnp.sum(particle_weights)) ** 2)
