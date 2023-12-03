from typing import Callable, List, Dict

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from pli.models.basic.uniform import UniformDistribution
from pli.models.basic.gaussian import GaussianMixtureDistribution

from .base import Task
from .utils import support_dict_to_array


class GaussianMixture(Task):
    name = "GaussianMixture"

    def __init__(
        self,
        n_train_data=50,
        n_eval_data=50,
        n_posterior_samples=1000,
        n_chains=1,
        n_warmup=1000,
        **_ignore
    ):
        super().__init__(
            n_train_data,
            n_eval_data,
            n_posterior_samples,
            n_chains,
            n_warmup,
        )

    def get_simulator(self) -> Callable:
        def simulator(rng_key, param) -> jnp.ndarray:
            # def mixture_sampler(rng_key, mean):
            mixture_coeffs = jnp.array([0.5, 0.5])
            means = jnp.tile(param[jnp.newaxis], (2, 1))
            covs = jnp.stack([jnp.eye(2), 0.01 * jnp.eye(2)])
            return GaussianMixtureDistribution(
                mixture_coeffs, means, covs
            ).sample(rng_key)[jnp.newaxis]

        return simulator

    def get_model(self) -> Callable:
        param_support = support_dict_to_array(self.param_support(), self.param_names())

        def model(obs: jnp.ndarray):
            # Bring observation into correct form
            obs = jnp.asarray(obs)

            params = numpyro.sample(
                "params", dist.Uniform(param_support[0], param_support[1])
            )

            # Define mixture distribution
            means = jnp.tile(params[jnp.newaxis], (2, 1))
            covs = jnp.stack([jnp.eye(2), 0.01 * jnp.eye(2)])
            mixing_dist = dist.Categorical(probs=jnp.ones(2) / 2.)
            component_dist = dist.MultivariateNormal(loc=means, covariance_matrix=covs)
            mixture = dist.MixtureSameFamily(mixing_dist, component_dist)
            return numpyro.sample(
                "obs",
                mixture,
                obs=obs[:, 0, :],
                sample_shape=(obs.shape[0], ),
            )

        return model

    def get_prior(self):
        param_support = support_dict_to_array(self.param_support(), self.param_names())
        return UniformDistribution(min_val=param_support[0], max_val=param_support[1])

    def param_names(self) -> List[str]:
        return ["xi_1", "xi_2"]

    def param_support(self) -> Dict:
        return {
            "xi_1": (-10.0, 10.0),
            "xi_2": (-10.0, 10.0),
        }

    @property
    def param_dim(self) -> int:
        return 2

    @property
    def data_dim(self) -> int:
        return 2

    def ground_truth_parameters(self) -> jnp.ndarray:
        return jnp.array([-9.527071, -1.4817104])
