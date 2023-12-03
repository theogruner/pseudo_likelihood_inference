from typing import Dict, Tuple

import jax.numpy as jnp
from jax.scipy.optimize import minimize

from pli.inference.pseudo_likelihood_inference.types import PLITrainState
from pli.models.density_estimators.gaussian import chol_to_cov, _log_prob, cov_to_chol


def build_gaussian_update(update_method="m-projection", **_ignore):
    """Optimize the Gaussian either via the i- or m-projection."""
    if update_method == "i-projection":
        update_method = update_i_projection
    elif update_method == "m-projection":
        update_method = update_m_projection
    else:
        raise ValueError(
            "Either `i`- or `m`-projection are supported as update strategy, "
            f"but {update_method} was given."
        )
    return update_method


def update_m_projection(
    params, likelihood: jnp.ndarray, train_state: PLITrainState
) -> Tuple[PLITrainState, Dict]:
    """Update the Gaussian parameters based on a closed form solution
    of the m-projection."""
    mean = jnp.mean(
        jnp.tile(likelihood[:, jnp.newaxis], (1, params.shape[-1])) * params, axis=0
    )

    # Update covariance
    mean_dist = params - jnp.tile(mean, (params.shape[0], 1))
    cov = jnp.cov(mean_dist, aweights=likelihood, rowvar=False) + 1e-6 * jnp.eye(
        params.shape[-1]
    )
    cholesky = cov_to_chol(cov)
    return (
        train_state.replace(
            model_params={"mean": mean, "cholesky": cholesky},
            model_update_steps=train_state.model_update_steps + 1,
        ),
        {},
    )


def update_i_projection(
    params, likelihood, train_state: PLITrainState
) -> Tuple[PLITrainState, Dict]:
    """Update the Gaussian parameters with numerical optimization based on
    the i-projection."""
    mean = train_state.model_params["mean"]
    cholesky = train_state.model_params["cholesky"]
    covariance = chol_to_cov(cholesky)
    gaussian_dim = mean.shape[-1]

    diag_idx = jnp.diag_indices(gaussian_dim)
    tril_idx = jnp.tril_indices(gaussian_dim, -1)

    log_likelihood = jnp.log(likelihood)

    res = minimize(
        primal_fun,
        x0=jnp.r_[mean, jnp.log(cholesky[diag_idx]), cholesky[tril_idx]],
        args=(mean, covariance, params, log_likelihood, diag_idx, tril_idx),
        method="BFGS",
        tol=1e-8,
        # options=dict(maxiter=100)
    )
    cholesky = jnp.zeros((gaussian_dim, gaussian_dim))
    cholesky = cholesky.at[diag_idx].set(
        jnp.exp(res.x[gaussian_dim : 2 * gaussian_dim])
    )
    cholesky = cholesky.at[tril_idx].set(res.x[2 * gaussian_dim :])
    return (
        train_state.replace(
            model_params={"mean": res.x[:gaussian_dim], "cholesky": cholesky},
            model_update_steps=train_state.model_update_steps + 1,
        ),
        {},
    )


def primal_fun(
    x, proposal_mean, proposal_covariance, params, log_likelihood, diag_idx, tril_idx
) -> jnp.ndarray:
    """Importance weighted objective of the i-projection"""
    gaussian_dim = proposal_mean.shape[-1]
    mean = x[:gaussian_dim]
    cholesky = jnp.zeros((gaussian_dim, gaussian_dim))
    cholesky = cholesky.at[diag_idx].set(jnp.exp(x[gaussian_dim : 2 * gaussian_dim]))
    cholesky = cholesky.at[tril_idx].set(x[2 * gaussian_dim :])
    cov = chol_to_cov(cholesky)
    log_prob = _log_prob(params, mean, cov)
    proposal_log_prob = _log_prob(params, proposal_mean, proposal_covariance)
    importance_ratio = jnp.exp(log_prob - proposal_log_prob)
    return -jnp.mean(importance_ratio * log_likelihood) + _kl(
        mean, cov, proposal_mean, proposal_covariance
    )


def _kl(q_mean, q_cov, p_mean, p_cov):
    """Kullback-Leibler divergence between two multivariate Gaussians q and p"""
    p_precision = jnp.linalg.inv(p_cov)
    trace = jnp.trace(p_precision @ q_cov)
    mean_diff = p_mean - q_mean
    p_sign, p_log_det = jnp.linalg.slogdet(p_cov)
    q_sign, q_log_det = jnp.linalg.slogdet(q_cov)
    return 0.5 * (
        trace
        + mean_diff.transpose() @ (p_precision @ mean_diff)
        + p_sign * p_log_det
        - q_sign * q_log_det
        - q_mean.shape[0]
    )
