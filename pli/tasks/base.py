import logging
import os
from abc import abstractmethod
from typing import Callable, Dict, List, Optional

import math
import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array
import haiku as hk
import optax
from flax.training.train_state import TrainState
from numpyro.infer import NUTS, MCMC
from numpyro.infer.util import log_density

from pli.figures import PairplotFigure
from pli.tasks.utils import support_dict_to_array
from pli.utils.dataloaders import build_dataloader, Dataset
from pli.models.density_estimators.nsf import forward_fn_log_prob, forward_fn_sample
from pli.utils.sampling import sample_within_support
from pli.utils.simulating import n_sims_per_param_wrapper


class Task:
    name = "Task"

    def __init__(
        self,
        n_train_data=50,
        n_eval_data=50,
        n_posterior_samples=1000,
        n_chains=1,
        n_warmup=1000,
        **_ignore,
    ):
        """
        :param n_train_data: Number of training data points.
        :param n_eval_data: Number of evaluation data points.
        :param n_posterior_samples: Number of posterior samples to collect.
        :param n_chains: Number of MCMC chains to run to collect the posterior data.
        :param n_warmup: Number of warm-up steps to take before
        collecting samples from the MCMC strategy.
        """
        self.n_train_data = n_train_data
        self.n_eval_data = n_eval_data
        self.n_posterior_samples = n_posterior_samples
        self.n_chains = n_chains
        self.n_warmup = n_warmup

    @abstractmethod
    def get_prior(self):
        """The prior of the assumed task."""

    @abstractmethod
    def get_simulator(self) -> Callable:
        """The simulator to draw samples from."""

    @abstractmethod
    def ground_truth_parameters(self) -> jnp.ndarray:
        """Predefined set of ground truth parameters."""

    def generate_reference_data(
        self,
        rng_key: Array,
        n_samples: int,
        data_dir=None,
        file_name="reference_data.npy",
    ) -> np.ndarray:
        """Generate reference data with Dim[n_reference_data, seq_len, data_dim]"""
        simulator = self.get_simulator()
        sim_wrapper = n_sims_per_param_wrapper(simulator, n_samples)
        simulations = sim_wrapper(rng_key, self.ground_truth_parameters())
        if data_dir is not None:
            logging.info(f"Storing reference data at {data_dir}")
            os.makedirs(data_dir, exist_ok=True)
            np.save(os.path.join(data_dir, file_name), simulations)
        return simulations

    @abstractmethod
    def param_support(self) -> Dict:
        """Defines the param distribution's support/range."""

    @abstractmethod
    def param_names(self) -> List[str]:
        """Define the relevant system parameter"""

    @property
    @abstractmethod
    def param_dim(self) -> int:
        """Dimension of the system parameters."""

    @property
    @abstractmethod
    def data_dim(self) -> int:
        """Dimension of the simulated data"""

    def initialize_task(self, data_dir, seed=None):
        """Initializes the task for LFI. All outputs are in numpy arrays.
        :param data_dir: Directory where the data should be saved
        :param seed: Optional seed to set for the
        """

        # Check whether `data_dir` already exists.
        os.makedirs(data_dir, exist_ok=True)

        if seed is None:
            seed = 42

        rng_key = jax.random.PRNGKey(seed)
        train_key, eval_key, posterior_key = jax.random.split(rng_key, 3)

        # Generate training and evaluation data
        if os.path.isdir(os.path.join(data_dir, "train")):
            train_data = np.load(os.path.join(data_dir, "train", "reference_data.npy"))
        else:
            train_data = self.generate_reference_data(
                train_key, self.n_train_data, os.path.join(data_dir, "train")
            )
        if os.path.isdir(os.path.join(data_dir, "eval")):
            eval_data = np.load(os.path.join(data_dir, "eval", "reference_data.npy"))
        else:
            eval_data = self.generate_reference_data(
                eval_key, self.n_eval_data, os.path.join(data_dir, "eval")
            )

        # Sample from posterior if available
        param_support = support_dict_to_array(self.param_support(), self.param_names())
        posterior_samples = None
        reference_posterior_path = os.path.join(
            data_dir, "reference_posterior", "reference_posterior_samples.npy"
        )

        if os.path.isfile(reference_posterior_path):
            posterior_samples = np.load(reference_posterior_path)
        elif self.get_posterior() is not None:
            os.makedirs(os.path.join(data_dir, "reference_posterior"), exist_ok=True)
            logging.info(
                f"Drawing {self.n_posterior_samples} posterior samples from analytical posterior."
            )
            posterior = self.get_posterior()(eval_data)
            posterior_samples = sample_within_support(
                posterior_key, self.n_posterior_samples, posterior.sample, param_support
            )
            np.save(
                reference_posterior_path,
                posterior_samples,
            )

        return (
            self.param_names(),
            self.get_prior(),
            self.get_simulator(),
            posterior_samples,
            jnp.asarray(param_support),
            train_data,
            eval_data,
        )

    def get_model(self) -> Optional[Callable]:
        """Returns a numpyro compatible model to generate posterior samples with MCMC.
        A model should look like this:

        def model(data: jnp.ndarray):
            # data the observed reference data with Dim[N, seq_len, D]
            # N: Batch size
            # seq_len: Sequence length
            # D: Dimensionality of the data
        """
        return None

    def get_posterior(self) -> Optional[Callable]:
        """Return an analytically derived posterior distribution."""
        return None

    @staticmethod
    def load_reference_data(data_dir):
        train_data = np.load(os.path.join(data_dir, "train", "reference_data.npy"))
        eval_data = np.load(os.path.join(data_dir, "eval", "reference_data.npy"))
        return train_data, eval_data

    @staticmethod
    def save_reference_data(data: np.ndarray, data_dir):
        np.save(os.path.join(data_dir, "reference_data.npy"), data)

    @property
    def data_dir(self) -> str:
        data_dir = f"{self.name.lower().replace('-', '_')}"
        if hasattr(self, "seq_len"):
            data_dir += f"-seq_len_{self.seq_len}"
        return data_dir + f"-n_train_samples_{self.n_train_data}"

    def task_specific_plots(self) -> List:
        return [PairplotFigure(self.param_names(), self.param_support())]
