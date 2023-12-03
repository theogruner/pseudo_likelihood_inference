from typing import List, Dict, Optional, Callable

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5

from pli.figures import TrajectoryFigure
from pli.models.basic.gaussian import LogNormalDistribution

from .base import Task


class SIR(Task):
    """Epidemiological time-series model for predicting the spreading of a disease.
    `S` denotes susceptible, `I` is an infectious agent,
    and `R` is a recovered or a deceased individual.
    """

    name = "SIR"

    def __init__(
        self,
        n_train_data: int = 50,
        n_eval_data: int = 50,
        n_posterior_samples: int = 1000,
        n_chains: int = 1,
        n_warmup: int = 1000,
        seq_len: int = 160,
        population_size: int = 1_000_000,
        observation_interval: int = 10,
        **_ignore,
    ):
        """
        :param seed: Set seeding for the stochastic simulator.
        :param n_train_data:
        :param n_eval_data:
        :param n_chains: Number of MCMC chains to run to collect the posterior data.
        :param n_warmup: Number of warm-up steps to take before
        collecting samples from the MCMC strategy.
        :param seq_len: Number of days that the disease threatens the population.
        :param population_size: The initial population size.
        :param observation_interval: The state disease is observed in a fixed interval days.
        """
        super().__init__(
            n_train_data, n_eval_data, n_posterior_samples, n_chains, n_warmup
        )
        self.seq_len = seq_len
        self.population_size = population_size
        self.observation_interval = observation_interval

    def get_prior(self):
        mean = jnp.log(jnp.array([0.4, 0.125]))
        var = jnp.array([0.5, 0.2])
        return LogNormalDistribution(mean, var)

    def get_simulator(self) -> Callable:
        def simulator(next_rng_key, param: np.ndarray) -> np.ndarray:
            """Simulate one disease case. Observations are taken at discrete time stamps."""
            population_size = self.population_size
            y0 = jnp.array([self.population_size - 1, 1, 0])
            t = jnp.linspace(0, self.seq_len, self.observation_interval + 1)[1:]

            def df(t, y, args):
                """
                :param t:
                :param y:
                :param n:
                :
                """
                ds_dt = -param[0] * y[0] * y[1] / population_size
                di_dt = (
                        param[0] * y[0] * y[1] / population_size
                        - param[1] * y[1]
                )
                dr_dt = param[1] * y[1]
                return jnp.stack([ds_dt, di_dt, dr_dt], axis=-1)

            term = ODETerm(df)
            solver = Tsit5()
            sol = diffeqsolve(
                term,
                solver,
                t0=0,
                t1=self.seq_len,
                dt0=1.0,
                y0=y0,
                saveat=SaveAt(ts=t),
            )
            bernoulli = numpyro.distributions.Binomial(
                1000, sol.ys.at[:, 1].get() / population_size
            )
            rng_key, next_rng_key = jax.random.split(next_rng_key)
            sample = bernoulli.sample(rng_key, sample_shape=())
            return sample[..., jnp.newaxis]
        return simulator

    def get_model(self) -> Callable:
        mean = jnp.log(jnp.array([0.4, 0.125]))
        cov = jnp.array([0.5, 0.2])
        # cov = jnp.array([[0.5, 0], [0.0, 0.2]])

        def model(obs: jnp.ndarray):

            obs = jnp.asarray(obs)
            params = numpyro.sample(
                "params", dist.LogNormal(mean, cov)
            )

            y0 = jnp.array([self.population_size - 1, 1, 0])
            t = jnp.linspace(0, self.seq_len, self.observation_interval + 1)[1:]

            @jax.jit
            def df(t, y, args):
                """
                :param t:
                :param y:
                :param args:
                :
                """
                ds_dt = -params[0] * y[0] * y[1] / self.population_size
                di_dt = (
                    params[0] * y[0] * y[1] / self.population_size
                    - params[1] * y[1]
                )
                dr_dt = params[1] * y[1]
                return jnp.stack([ds_dt, di_dt, dr_dt], axis=-1)

            term = ODETerm(df)
            solver = Tsit5()
            sol = diffeqsolve(
                term, solver, t0=0, t1=self.seq_len, dt0=1.0, y0=y0, saveat=SaveAt(ts=t)
            )
            return numpyro.sample(
                "obs",
                dist.Binomial(
                    1000,
                    sol.ys[:, 1][:, jnp.newaxis]
                    / self.population_size,
                ),
                obs=obs,
                sample_shape=(obs.shape[0],)
            )

        return model

    def ground_truth_parameters(self) -> np.ndarray:
        return np.array([0.6, 0.2])

    def param_support(self) -> Dict:
        return {"beta": (1e-6, 2.0), "gamma":(1e-6, 0.5)}

    def param_names(self) -> List[str]:
        """
        beta: Contact rate
        gamma: Mean recovery rate
        """
        return ["beta", "gamma"]

    @property
    def param_dim(self) -> int:
        return 2

    @property
    def data_dim(self) -> int:
        return 1

    def task_specific_plots(self) -> List:
        plots = super().task_specific_plots()
        return plots + [TrajectoryFigure(self.data_dim)]
