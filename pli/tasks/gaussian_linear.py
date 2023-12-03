from typing import List, Dict, Callable

import jax
import jax.numpy as jnp


from pli.models.basic.gaussian import GaussianDistribution
from .base import Task


class GaussianLinear(Task):
    name = "GaussianLinear"

    def __init__(
        self,
        n_train_data=50,
        n_eval_data=50,
        n_posterior_samples=1000,
        n_chains=1,
        n_warmup=1000,
        n_dim: int = 10,
        **_ignore,
    ):
        super().__init__(
            n_train_data, n_eval_data, n_posterior_samples, n_chains, n_warmup
        )
        self.n_dim = n_dim

    def get_prior(self):
        n = self.param_dim
        mean = jnp.zeros(n)
        cov = 0.1 * jnp.eye(n)
        return GaussianDistribution(mean, cov)

    def get_simulator(self) -> Callable:
        n = self.param_dim

        def simulator(rng_key, param: jnp.ndarray):
            cov = 0.1 * jnp.eye(n)
            return jax.random.multivariate_normal(rng_key, param, cov, shape=())[jnp.newaxis]

        return simulator

    def get_posterior(self):
        n = self.param_dim

        def posterior(observation):
            """
            See
            `https://math.stackexchange.com/questions/157172
            /product-of-two-multivariate-gaussians-distributions`
            for a product of multivariate Gaussians.
            """
            prior_mean = jnp.zeros(n)
            prior_precision = 10 * jnp.eye(n)
            likelihood_mean = observation.squeeze(1)
            likelihood_precision = 10 * jnp.eye(n)
            n_samples_per_param = observation.shape[0]

            posterior_cov = jnp.diag(
                1
                / (
                    jnp.diag(prior_precision)
                    + n_samples_per_param * jnp.diag(likelihood_precision)
                )
            )
            posterior_mean = posterior_cov @ (
                prior_precision @ prior_mean
                + jnp.sum(
                    (
                        jnp.tile(likelihood_precision, (n_samples_per_param, 1, 1))
                        @ likelihood_mean[..., jnp.newaxis]
                    ).squeeze(-1),
                    axis=0,
                )
            )
            return GaussianDistribution(posterior_mean, posterior_cov)

        return posterior

    def ground_truth_parameters(self) -> jnp.ndarray:
        """Sample `n_dim` parameters between [-1., 1.].
        The seed is fixed to ensure a deterministic choice of parameters."""
        if self.param_dim == 10:
            gt_params = jnp.array(
                [
                    -0.3833964,
                    -0.32583132,
                    -0.11903118,
                    -0.16812938,
                    -0.5765837,
                    -0.37973747,
                    0.33850512,
                    0.04474947,
                    -0.51732945,
                    -0.37251562,
                ]
            )
            return gt_params[:len(self.param_names())]
        seed = 42
        rng_key = jax.random.PRNGKey(seed)
        return jax.random.uniform(rng_key, shape=(self.param_dim, ), minval=-1., maxval=1.)

    def param_support(self) -> Dict:
        return {f"x_{i}": (-1.0, 1.0) for i in range(1, self.param_dim + 1)}

    def param_names(self) -> List[str]:
        return [f"x_{i}" for i in range(1, self.param_dim + 1)]

    @property
    def param_dim(self) -> int:
        return self.n_dim

    @property
    def data_dim(self) -> int:
        return self.param_dim

    @property
    def data_dir(self) -> str:
        data_dir = f"{self.name.lower().replace('-', '_')}"
        data_dir += f"-n_dim_{self.param_dim}"
        return data_dir + f"-n_train_samples_{self.n_train_data}"

    def task_specific_plots(self) -> List:
        plots = super().task_specific_plots()
        n_plotting_params = 5
        for plot in plots:
            if plot.name == "PairplotFigure":
                plot.n_plotting_params = n_plotting_params
        return plots
