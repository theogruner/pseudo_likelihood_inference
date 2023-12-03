from typing import Dict, List, Callable, Optional

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from pli.figures.task_specific import GaussianFigure
from pli.models.basic.uniform import UniformDistribution

from .utils import support_dict_to_array
from .base import Task


class SLCP(Task):
    name = "SLCP"

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
            n_train_data, n_eval_data, n_posterior_samples, n_chains, n_warmup
        )

    def ground_truth_parameters(self) -> np.ndarray:
        mu_1 = 0.7
        mu_2 = 1.5
        s_1 = -1.0
        s_2 = -0.9
        rho = 0.6
        return np.array([mu_1, mu_2, s_1, s_2, rho])

    def param_support(self) -> Dict:
        return {
            "xi_1": (-3.0, 3.0),
            "xi_2": (-3.0, 3.0),
            "xi_3": (-3.0, 3.0),
            "xi_4": (-3.0, 3.0),
            "xi_5": (-3.0, 3.0),
        }

    @property
    def param_dim(self):
        return 5

    @property
    def data_dim(self) -> int:
        return 8

    def param_names(self) -> List[str]:
        return ["xi_1", "xi_2", "xi_3", "xi_4", "xi_5"]

    def get_prior(self):
        param_support = support_dict_to_array(self.param_support(), self.param_names())
        return UniformDistribution(min_val=param_support[0], max_val=param_support[1])

    def get_simulator(self) -> Callable:
        def simulator(rng_key, param) -> jnp.ndarray:
            mean = param[:2]
            cov_non_diag = (jnp.tanh(param[4]) * (param[2] ** 2) * (param[3] ** 2))
            cov = jnp.array(
                [[param[2] ** 4, cov_non_diag], [cov_non_diag, param[3] ** 4]]
            ) + 1e-6 * jnp.eye(2)
            samples = jax.random.multivariate_normal(
                rng_key, mean, cov, shape=(4, )
            )
            return samples.reshape((-1, 8))

        return simulator

    def get_model(self) -> Optional[Callable]:
        lower_support = jnp.array(
            [self.param_support()[key][0] for key in self.param_names()]
        )
        upper_support = jnp.array(
            [self.param_support()[key][1] for key in self.param_names()]
        )

        # @jax.jit
        def model(obs: jnp.ndarray):
            # Bring observation into correct form
            obs = jnp.asarray(obs)
            obs = obs.squeeze(axis=1).reshape((obs.shape[0], 4, 2)) # .transpose(1, 0, 2)

            params = numpyro.sample(
                "params", dist.Uniform(lower_support, upper_support)
            )

            mean = params[:2]
            cov_non_diag = (
                jnp.tanh(params[4]) * (params[2] ** 2) * (params[3] ** 2)
            )
            cov = jnp.array([
                [params[2] ** 4, cov_non_diag],
                [cov_non_diag, params[3] ** 4]]
            )
            return numpyro.sample(
                "obs",
                dist.MultivariateNormal(mean, cov),
                obs=obs,
                sample_shape=(obs.shape[0], 4,),
            )

        return model

    def task_specific_plots(self) -> List:
        plots = super().task_specific_plots()
        return plots + [
            GaussianFigure(),
        ]
