from typing import Dict

import jax
import jax.numpy as jnp

from .types import PerturbationKernel
from .independant_kernel import weighted_covariance, weighted_variance


def build_conditional_gaussian_kernel():
    """Build a conditional Gaussian kernel with isotropic covariance."""
    def init_gaussian_kernel(rng_key, particles, weights):
        kernel_cov = weighted_variance(particles, weights)
        return {"kernel_cov": kernel_cov}

    return PerturbationKernel(
        init=init_gaussian_kernel,
        forward=conditional_gaussian_kernel_forward,
        log_prob=conditional_gaussian_kernel_log_prob,
        sample=conditional_gaussian_kernel_sample,
        update=conditional_gaussian_kernel_update,
    )


def build_multivariate_conditional_gaussian_kernel():
    """build a conditional multivariate Gaussian kernel with full cov matrix."""
    def init_multivariate_conditional_kernel(rng_key, particles, weights):
        kernel_cov = weighted_covariance(particles, weights)
        return {"kernel_cov": kernel_cov}

    return PerturbationKernel(
        init=init_multivariate_conditional_kernel,
        forward=conditional_gaussian_kernel_forward,
        log_prob=conditional_gaussian_kernel_log_prob,
        sample=conditional_gaussian_kernel_sample,
        update=multivariate_conditional_gaussian_kernel_update,
    )


def conditional_gaussian_kernel_forward(x, condition, kernel_params: Dict):
    x_reshaped = jnp.tile(x[:, jnp.newaxis, :], (1, condition.shape[0], 1))
    condition_reshaped = jnp.tile(condition, (x.shape[0], 1, 1))
    kernel_cov_reshaped = jnp.tile(
        kernel_params["kernel_cov"], (x.shape[0], condition.shape[0], 1, 1)
    )
    return jax.scipy.stats.multivariate_normal.pdf(
        x_reshaped, condition_reshaped, kernel_cov_reshaped  # kernel_params["kernel_cov"]
    )


def conditional_gaussian_kernel_log_prob(x, condition, kernel_params: Dict):
    return jax.scipy.special.logsumexp(
        jnp.log(conditional_gaussian_kernel_forward(x, condition, kernel_params)),
        axis=-1,
    )


def conditional_gaussian_kernel_sample(rng_key, particles, kernel_params: Dict):
    return jax.random.multivariate_normal(
        key=rng_key,
        mean=particles,
        cov=kernel_params["kernel_cov"],
    )


def conditional_gaussian_kernel_update(rng_key, particles, weights):
    kernel_cov = weighted_variance(particles, weights)
    return {"kernel_cov": kernel_cov}


def multivariate_conditional_gaussian_kernel_update(rng_key, particles, weights):
    kernel_cov = weighted_covariance(particles, weights)
    return {"kernel_cov": kernel_cov}
