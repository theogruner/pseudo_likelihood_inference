"""One-step updates of Markov-chain sampling strategies."""
from typing import Dict, Any, Optional

import jax
import jax.numpy as jnp

from .markov_kernels.kernels import build_markov_kernel
from .types import ParticleSampler


def build_smc_sampler(
    param_support: jnp.ndarray,
    markov_kernel: Optional[str] = None,
    **_ignore
):
    """Build a particle sampler"""
    markov_kernel = build_markov_kernel(markov_kernel)

    def init_particle_sampler(rng_key: jnp.ndarray, n_param_samples) -> Dict[str, Any]:
        particles = jax.random.uniform(
            rng_key,
            shape=(n_param_samples, param_support[0].size),
            minval=param_support[0],
            maxval=param_support[1],
        )
        weights = jnp.ones(particles.shape[0]) / particles.shape[0]
        kernel_params = markov_kernel.init(rng_key, particles, weights)
        return {"particles": particles, "weights": weights, "kernel_params": kernel_params}

    return ParticleSampler(
        init=init_particle_sampler,
        sample=lambda rng, sample_shape, train_state: sample(
            rng, sample_shape, train_state, markov_kernel
        ),
        log_prob=lambda x, model_params: log_prob(x, model_params, markov_kernel),
        pdf=lambda x, model_params: pdf(x, model_params, markov_kernel),
    )


def sample(rng_key, sample_shape, model_params: Dict, markov_kernel):
    particles = model_params["particles"]
    weights = model_params["weights"]
    rng_key1, rng_key2 = jax.random.split(rng_key, 2)
    particle_idx = jax.random.categorical(
        rng_key1, jnp.log(weights), shape=sample_shape
    )
    return markov_kernel.sample(
        rng_key2, particles[particle_idx], model_params["kernel_params"]
    )


def log_prob(x, model_params: Dict, markov_kernel):
    return markov_kernel.log_prob(
        x, model_params["particles"], model_params["kernel_params"]
    )


def pdf(x, model_params: Dict, markov_kernel):
    return log_prob(x, model_params, markov_kernel)
