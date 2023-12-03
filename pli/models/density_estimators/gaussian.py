from typing import Dict, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array

from .types import DensityEstimator


def build_gaussian(param_support: Optional[Array] = None, n_dim=None, **_ignore):
    """Build a Gaussian density estimator. """
    if param_support is None and n_dim is None:
        raise ValueError("Gaussian Model requires a support region to be initialized")
    if param_support is None:
        # Center Gaussian around 0
        param_support = jnp.tile(jnp.array([[-1.], [1.]]), (1, n_dim))

    def init_gaussian(rng_key: Array) -> Dict[str, Array]:
        mean = jnp.asarray((param_support[0] + param_support[1]) / 2)
        covariance = jnp.diag(jnp.asarray((param_support[1] - param_support[0]) / 10))
        cholesky = cov_to_chol(covariance)
        model_params = {"mean": mean, "cholesky": cholesky}
        return model_params

    def sample(rng_key: Array, sample_shape, model_params: Dict):
        mean = model_params["mean"]
        cov = chol_to_cov(model_params["cholesky"])
        return jax.random.multivariate_normal(rng_key, mean, cov, sample_shape)

    def log_prob(x: Array, model_params: Dict):
        mean = model_params["mean"]
        cholesky = model_params["cholesky"]
        return _log_prob(x, mean, chol_to_cov(cholesky))

    def pdf(x: Array, model_params: Dict):
        return log_prob(x, model_params)

    return DensityEstimator(
        init=init_gaussian, sample=sample, log_prob=log_prob, pdf=pdf
    )


def _log_prob(x: Array, mean: Array, cov: Array):
    x = jnp.atleast_2d(x)

    mean_dist = (x - mean)[..., jnp.newaxis]
    exp_term = -0.5 * mean_dist.transpose(0, 2, 1) @ (jnp.linalg.inv(cov) @ mean_dist)
    sign, log_det = jnp.linalg.slogdet(cov)
    log_prob_const = -0.5 * (
        jnp.log(2 * jnp.pi) * mean.shape[0] + sign * log_det
    )
    return (exp_term + log_prob_const).squeeze()


def chol_to_cov(cholesky: Array):
    return cholesky @ cholesky.transpose()


def cov_to_chol(covariance: Array):
    return jnp.linalg.cholesky(covariance)
