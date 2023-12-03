from functools import partial
from typing import Dict

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.scipy.linalg import solve_triangular

from .types import PerturbationKernel


def build_multivariate_gaussian_kernel():
    """Build a multivariate Gaussian kernel with full cov matrix"""
    def init_multivariate_gaussian_kernel(rng_key, particles, weights):
        kernel_mean = jnp.mean(particles, axis=0)
        kernel_cov = weighted_covariance(particles, weights)
        return {"kernel_mean": kernel_mean, "kernel_cov": kernel_cov}

    return PerturbationKernel(
        init=init_multivariate_gaussian_kernel,
        forward=multivariate_gaussian_kernel_forward,
        log_prob=multivariate_gaussian_kernel_logpdf,
        sample=multivariate_gaussian_kernel_sample,
        update=update_multivariate_gaussian_kernel,
    )


def build_gaussian_kernel():
    """Build a Gaussian kernel with isotropic cov matrix"""
    def init_gaussian_kernel(rng_key, particles, weights):
        kernel_mean = jnp.mean(particles, axis=0)
        kernel_cov = weighted_variance(particles, weights)
        return {"kernel_mean": kernel_mean, "kernel_cov": kernel_cov}

    return PerturbationKernel(
        init=init_gaussian_kernel,
        forward=multivariate_gaussian_kernel_forward,
        log_prob=multivariate_gaussian_kernel_logpdf,
        sample=multivariate_gaussian_kernel_sample,
        update=update_gaussian_kernel,
    )


def multivariate_gaussian_kernel_sample(rng_key, particles, kernel_params: Dict):
    return jax.random.multivariate_normal(
        key=rng_key,
        mean=kernel_params["kernel_mean"],
        cov=kernel_params["kernel_cov"],
        shape=(particles.shape[0],),
    )


def multivariate_gaussian_kernel_forward(x, condition, kernel_params: Dict):
    x_reshaped = jnp.tile(x[:, jnp.newaxis, :], (1, condition.shape[0], 1))
    mean_reshaped = jnp.tile(
        kernel_params["kernel_mean"], (x.shape[0], condition.shape[0], 1)
    )
    return jax.scipy.stats.multivariate_normal.pdf(
        x_reshaped, mean_reshaped, kernel_params["kernel_cov"]
    )


def multivariate_gaussian_kernel_logpdf(x, condition, kernel_params: Dict):
    return jax.scipy.stats.multivariate_normal.logpdf(
        x, kernel_params["kernel_mean"], kernel_params["kernel_cov"]
    )


def weighted_covariance(particles: jnp.ndarray, weights: jnp.ndarray):
    return jnp.cov(particles, rowvar=False, aweights=weights) + 1e-6 * jnp.diag(
        jnp.ones(particles.shape[-1])
    )


def weighted_variance(particles, weights):
    weighted_average = jnp.average(particles, weights=weights, axis=0)
    return jnp.diag(
        jnp.average((particles - weighted_average) ** 2, weights=weights, axis=0) + 1e-6
    )


def update_multivariate_gaussian_kernel(rng_key, particles, weights):
    kernel_mean = jnp.mean(particles, axis=0)
    kernel_cov = weighted_covariance(particles, weights)
    return {"kernel_mean": kernel_mean, "kernel_cov": kernel_cov}


def update_gaussian_kernel(rng_key, particles, weights):
    kernel_mean = jnp.mean(particles, axis=0)
    kernel_cov = weighted_variance(particles, weights)
    return {"kernel_mean": kernel_mean, "kernel_cov": kernel_cov}


def build_gmm_kernel(k=5):
    def update_fn(rng_key, particles, weights):
        return update_gmm_kernel(rng_key, particles, k)

    return PerturbationKernel(
        init=update_fn,
        forward=gmm_kernel_forward,
        log_prob=gmm_kernel_log_pdf,
        sample=gmm_kernel_sample,
        update=update_fn,
    )


def gmm_kernel_forward(x, condition, kernel_params: Dict):
    log_probs = gmm_kernel_log_pdf(x, condition, kernel_params)
    return jnp.tile(log_probs[:, jnp.newaxis], (1, condition.shape[0]))


def gmm_kernel_log_pdf(x, condition, kernel_params: Dict):
    pi = kernel_params["kernel_coeffs"]
    mu = kernel_params["kernel_means"]
    precision_chol = kernel_params["kernel_precision_chols"]
    return logsumexp(jnp.log(pi) + _log_prob_normal(x, mu, precision_chol), axis=-1)


def _log_prob_normal(x, mu, precision_chol):
    """Calculate the pdf of a normal distribution with mean `mu_i` and co-variance `Sigma_i`.

    Args:
        x: Sample of Dim[n, d]

    Returns:

    """
    precision = _calc_precision(precision_chol)

    mean_dist = jnp.tile(
        x[:, jnp.newaxis, :, jnp.newaxis], (1, mu.shape[0], 1, 1)
    ) - jnp.tile(mu[jnp.newaxis, :, :, jnp.newaxis], (x.shape[0], 1, 1, 1))
    exponential_term = mean_dist.transpose((0, 1, 3, 2)) @ (
        jnp.tile(precision, (x.shape[0], 1, 1, 1)) @ mean_dist
    )
    sign, logdet = jnp.linalg.slogdet(precision)
    log_prob_const = jnp.log(2 * jnp.pi) * x.shape[-1] - sign * logdet
    return -0.5 * (exponential_term.squeeze(axis=(-1, -2)) + log_prob_const)


def _calc_precision(precision_chol):
    return precision_chol @ precision_chol.swapaxes(-1, -2)


def gmm_kernel_sample(rng_key, particles, kernel_params: Dict):
    rng_key1, rng_key2 = jax.random.split(rng_key)
    mu = kernel_params["kernel_means"]
    precision_chol = kernel_params["kernel_precision_chols"]
    pi = kernel_params["kernel_coeffs"]
    components_idx = jax.random.categorical(
        rng_key1, logits=jnp.log(pi), shape=(particles.shape[0],)
    )
    # See
    noise = solve_triangular(
        precision_chol[components_idx].transpose((0, 2, 1)),
        jax.random.normal(rng_key2, shape=(particles.shape[0], mu.shape[-1])),
    )
    return mu[components_idx] + noise


def update_gmm_kernel(rng_key, particles, k):
    """Highly inspired by this amazing colab
    https://colab.research.google.com/drive/1AwS4haUx6swF82w3nXr6QKhajdF8aSvA """
    kernel_means, _ = kmeans(rng_key, particles, k, 1)
    assignments, _ = vector_quantize(particles, kernel_means)

    kernel_coeffs = gmm_coeffs(assignments, k)

    kernel_covs = gmm_covs(particles, assignments, k)
    kernel_precisions = jnp.linalg.inv(kernel_covs)
    kernel_precision_chols = jnp.linalg.cholesky(kernel_precisions).transpose((0, 2, 1))
    return {
        "kernel_coeffs": kernel_coeffs,
        "kernel_means": kernel_means,
        "kernel_precision_chols": kernel_precision_chols
    }


@partial(jax.jit, static_argnums=(2, 3))
def kmeans(key, points, k, restarts, **kwargs):
    all_centroids, all_distortions = jax.vmap(
        lambda key: kmeans_run(key, points, k, **kwargs)
    )(jax.random.split(key, restarts))
    i = jnp.argmin(all_distortions)
    return all_centroids[i], all_distortions[i]


def gmm_coeffs(assignments, k):
    def calc_coeffs(i, coeffs):
        return coeffs.at[i].set(jnp.sum(assignments == i))

    pi = jnp.zeros(k)
    pi = jax.lax.fori_loop(0, k, calc_coeffs, pi)

    return pi / assignments.shape[0]


def gmm_covs(points, assignments, k):
    points_dim = points.shape[-1]
    covs = jnp.zeros((k, points_dim, points_dim))

    def fill_cov(i, c):
        p = jnp.where(
            jnp.tile(
                (assignments == i)[:, jnp.newaxis], (1, points.shape[-1])
            ), points, jnp.zeros_like(points)
        )
        weights = jnp.where(assignments == i, jnp.ones(points.shape[0]), jnp.zeros(points.shape[0]))
        # p = points.at[assignments == i].get()
        cov = jnp.cov(p, rowvar=False, aweights=weights) + 1e-6 * jnp.diag(jnp.ones(points_dim))
        return c.at[i].set(cov)

    covs = jax.lax.fori_loop(0, k, fill_cov, covs)

    return covs


def vector_quantize(points, codebook):
    assignment = jax.vmap(
        lambda point: jnp.argmin(jax.vmap(jnp.linalg.norm)(codebook - point))
    )(points)
    distns = jax.vmap(jnp.linalg.norm)(codebook[assignment, :] - points)
    return assignment, distns


@partial(jax.jit, static_argnums=(2,))
def kmeans_run(key, points, k, thresh=1e-5):
    def improve_centroids(val):
        prev_centroids, prev_distn, _ = val
        assignment, distortions = vector_quantize(points, prev_centroids)

        # Count number of points assigned per centroid
        counts = (
            (assignment[jnp.newaxis, :] == jnp.arange(k)[:, jnp.newaxis])
            .sum(axis=1, keepdims=True)
            .clip(min=1.0)  # clip to change 0/0 later to 0/1
        )

        # Sum over points in a centroid by zeroing others out
        new_centroids = (
            jnp.sum(
                jnp.where(
                    # axes: (data points, clusters, data dimension)
                    assignment[:, jnp.newaxis, jnp.newaxis]
                    == jnp.arange(k)[jnp.newaxis, :, jnp.newaxis],
                    points[:, jnp.newaxis, :],
                    0.0,
                ),
                axis=0,
            )
            / counts
        )

        return new_centroids, jnp.mean(distortions), prev_distn

    initial_indices = jax.random.shuffle(key, jnp.arange(points.shape[0]))[:k]
    initial_val = improve_centroids((points[initial_indices, :], jnp.inf, None))
    centroids, distortion, _ = jax.lax.while_loop(
        lambda val: (val[2] - val[1]) > thresh,
        improve_centroids,
        initial_val,
    )
    return centroids, distortion
