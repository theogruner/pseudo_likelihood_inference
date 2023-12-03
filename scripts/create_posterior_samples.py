"""Script to generate posterior samples with MCMC sampling
when the tasks model is given."""
import math
import os
import warnings
from typing import Callable, Union, Dict

import hydra
import numpy as np
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
import seaborn as sns
import pandas as pd

import jax
import jax.numpy as jnp
import jax.experimental.host_callback as hcb
import optax
from numpyro.infer import (
    NUTS,
    MCMC,
    init_to_median,
)
from numpyro.infer.util import log_density
import haiku as hk
from flax.training import checkpoints
from flax.training.train_state import TrainState

from pli.models.density_estimators.nsf import forward_fn_log_prob, forward_fn_sample
import pli.tasks
from pli.tasks import Task
from pli.tasks.utils import support_dict_to_array
from pli.utils.dataloaders import build_dataloader, Dataset
from pli.utils.sampling import sample_within_support


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    load_mcmc_proposals = True
    seed = cfg.method.seed
    next_rng_key = jax.random.PRNGKey(seed)

    if cfg.task.n_eval_data != cfg.task.n_train_data:
        warnings.warn(
            f"`n_eval_data` and `n_train_data` are different. "
            f"If this choice is on purpose, consider commenting this part out. "
            f"We set cfg.task.n_eval_data={cfg.task.n_train_data}."
        )
        cfg.task.n_eval_data = cfg.task.n_train_data

    # Load task
    task_ = getattr(pli.tasks, cfg.task.name)
    task: Task = task_(seed=seed, **cfg.task)

    model = task.get_model()
    if model is None:
        raise ValueError("There is no joint model to evaluate posterior samples.")

    param_support = support_dict_to_array(task.param_support(), task.param_names())

    # Set data directory
    data_dir = cfg.data_dir
    if data_dir is None:
        data_dir = os.path.join(pli.DATA_DIR, task.data_dir, f"seed_{seed}")

    (
        _,
        prior,
        _,
        posterior_samples,
        param_support,
        _,
        reference_data,
    ) = task.initialize_task(data_dir, seed)

    # We store all data and models in the directory `reference_posterior`
    reference_posterior_dir = os.path.join(data_dir, "reference_posterior")
    os.makedirs(reference_posterior_dir, exist_ok=True)
    ref_posterior_path = os.path.join(
        reference_posterior_dir, "reference_posterior_samples.npy"
    )
    if os.path.isfile(ref_posterior_path):
        print(
            f"Posterior samples already exist. If you want to calculate new samples, "
            f"please remove the old ones from:"
            f"\n{ref_posterior_path}"
        )
        plot_samples(reference_posterior_dir, "reference_posterior_samples")
        if cfg.plotting:
            plt.show()
        return

    with open(os.path.join(reference_posterior_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)

    rng_key, next_rng_key = jax.random.split(next_rng_key)

    # 1. Carry out inference with MCMC sampling
    mcmc_proposals_path = os.path.join(reference_posterior_dir, "mcmc_proposals.npy")
    if os.path.isfile(mcmc_proposals_path) and load_mcmc_proposals:
        mcmc_proposals = jnp.asarray(np.load(mcmc_proposals_path))
        print(
            f"Loaded {mcmc_proposals.shape[0]} proposal samples from "
            f"{mcmc_proposals_path}."
        )
    else:
        mcmc_proposals = run_mcmc(
            rng_key,
            model,
            reference_data,
            cfg.task.posterior_sampling.n_train_data,
            cfg.task.posterior_sampling.n_eval_data,
            cfg.task.posterior_sampling.n_warmup,
            cfg.task.posterior_sampling.n_chains,
        )
        print(f"Generated {mcmc_proposals.shape[0]} proposal samples.")
        np.save(
            os.path.join(mcmc_proposals_path),
            mcmc_proposals,
        )
        plot_samples(reference_posterior_dir, "mcmc_proposals")
        print(
            f"Saved {mcmc_proposals.shape[0]} proposal samples at {mcmc_proposals_path}."
        )

    # 2. Do density estimation on the proposal data

    # Initialize the model
    rng_key, next_rng_key = jax.random.split(next_rng_key)
    prior_samples = prior.sample(rng_key, (10_000,))
    t_mean = jnp.mean(prior_samples, axis=0)
    t_std = jnp.std(prior_samples, axis=0)
    t_std = jnp.where(t_std < 1e-14, 1e-14 * jnp.ones_like(t_std), t_std)
    forward_model_log_prob = forward_fn_log_prob(
        (param_support.shape[-1],),
        cfg.task.posterior_sampling.num_layers,
        cfg.task.posterior_sampling.hidden_size,
        cfg.task.posterior_sampling.mlp_num_layers,
        cfg.task.posterior_sampling.num_bins,
        t_mean,
        t_std,
    )
    sample_data = jnp.zeros((1, param_support.shape[-1]))
    model_params = forward_model_log_prob.init(rng_key, sample_data)
    optimizer = optax.adam(cfg.task.posterior_sampling.lr)
    train_state = TrainState.create(
        apply_fn=forward_model_log_prob.apply, params=model_params, tx=optimizer
    )

    # Load or train the model
    if os.path.isfile(os.path.join(reference_posterior_dir, "checkpoint_0")):
        train_state = checkpoints.restore_checkpoint(
            reference_posterior_dir, train_state
        )
        print(
            f"Loaded train state from {os.path.join(reference_posterior_dir, 'checkpoint_0')}"
        )
    else:
        rng_key, next_rng_key = jax.random.split(next_rng_key)
        train_state = run_density_estimation(
            rng_key,
            train_state,
            mcmc_proposals,
            cfg.task.posterior_sampling.n_train_data,
            cfg.task.posterior_sampling.training_steps,
            cfg.task.posterior_sampling.batch_size,
            cfg.task.posterior_sampling.eval_frequency,
        )

        checkpoints.save_checkpoint(
            reference_posterior_dir, train_state, 0, overwrite=True
        )

    forward_model_sample = forward_fn_sample(
        (param_support.shape[-1],),
        cfg.task.posterior_sampling.num_layers,
        cfg.task.posterior_sampling.hidden_size,
        cfg.task.posterior_sampling.mlp_num_layers,
        cfg.task.posterior_sampling.num_bins,
        t_mean,
        t_std,
    )

    # 3. Define a mixture distribution between the density estimator and the prior.
    print(
        "Define mixture density between the prior and the learned density estimator"
        f" with mixture coefficients {cfg.task.posterior_sampling.prior_mixture_coeff} "
        f"and {1 - cfg.task.posterior_sampling.prior_mixture_coeff}."
    )
    mixture_coeffs = jnp.array(
        [
            cfg.task.posterior_sampling.prior_mixture_coeff,
            1 - cfg.task.posterior_sampling.prior_mixture_coeff,
        ]
    )

    def mixture_sampler(rng_key, sample_shape):
        """Sampling utility for the mixture sampler"""
        rng_keys = jax.random.split(rng_key, sample_shape[0] + 1)
        rng_keys, rng_key1 = jnp.split(rng_keys, [sample_shape[0]])

        # Decide whether the prior or the nde should be used to sample
        mixture_idx = jax.random.categorical(
            rng_key1.squeeze(0),
            logits=jnp.log(mixture_coeffs),
            shape=sample_shape,
        ).astype(jnp.bool_)

        # Sample one realization from the mixture density
        def sample(mixture_comp, rng_key):
            return jax.lax.cond(
                mixture_comp,
                lambda rng_key: forward_model_sample.apply(
                    train_state.params, rng_key, (1,)
                ).squeeze(0),
                lambda rng_key: prior.sample(rng_key, (1,)).squeeze(0),
                rng_key,
            )

        # Sample n realizations
        return jax.vmap(
            sample,
            (0, 0),
            0,
        )(mixture_idx, rng_keys)

    def mixture_log_prob_fn(sample):
        """Evaluate the log prob of the mixture distribution"""
        log_prior = prior.log_prob(sample)
        log_density_estimator = train_state.apply_fn(train_state.params, sample)
        log_proposal = jax.scipy.special.logsumexp(
            jnp.stack(
                [
                    jnp.log(mixture_coeffs[0]) + log_prior,
                    jnp.log(mixture_coeffs[1]) + log_density_estimator,
                ]
            ),
            axis=0,
        )
        return log_proposal

    # 4. Carry out rejection sampling with the proposal density estimator
    rng_key, next_rng_key = jax.random.split(next_rng_key)
    posterior_samples = run_rejection_sampling(
        rng_key,
        cfg.task.posterior_sampling.n_posterior_samples,
        model,
        prior,
        mixture_sampler,
        mixture_log_prob_fn,
        reference_data,
        param_support,
    )

    print(f"Generated {posterior_samples.shape[0]} posterior samples.")
    np.save(
        os.path.join(reference_posterior_dir, "reference_posterior_samples.npy"),
        posterior_samples,
    )
    print(
        f"Saved {posterior_samples.shape[0]} proposal samples "
        f"at {os.path.join(reference_posterior_dir, 'reference_posterior_samples.npy')}."
    )
    plot_samples(reference_posterior_dir, "reference_posterior_samples")

    if cfg.plotting:
        plt.show()


def run_mcmc(
    rng_key,
    model: Callable,
    reference_data: Union[np.ndarray, jnp.ndarray],
    n_train_data: int,
    n_eval_data: int,
    n_warmup: int = 1000,
    n_chains: int = 1,
) -> jnp.ndarray:
    """Running initial MCMC sampling"""

    n_proposal_samples_per_chain = math.ceil((n_train_data + n_eval_data) / n_chains)
    kernel = NUTS(
        model,
        dense_mass=False,
        init_strategy=init_to_median(num_samples=10_000),
        target_accept_prob=0.7,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=n_warmup,
        num_samples=n_proposal_samples_per_chain,
        num_chains=n_chains,
        chain_method="parallel",
    )
    print(
        f"Running initial MCMC sampling with {n_chains} chains "
        f"for {n_proposal_samples_per_chain} prior samples after {n_warmup} warmup steps."
    )
    mcmc.run(rng_key, reference_data, extra_fields=["accept_prob"])

    return mcmc.get_samples()["params"]


def run_density_estimation(
    next_rng_key,
    state: TrainState,
    proposal_samples,
    n_train_data,
    n_training_steps,
    batch_size,
    eval_frequency,
) -> TrainState:
    """Running density estimation of the posterior"""
    # Create datasets for training and evaluation
    train_data = proposal_samples.at[:n_train_data].get()
    eval_data = proposal_samples.at[n_train_data:].get()
    train_dataset = Dataset.create(train_data)
    eval_dataset = Dataset.create(eval_data)

    # Create the required dataloaders
    train_dataloader = build_dataloader(train_dataset, batch_size=batch_size)
    eval_dataloader = build_dataloader(eval_dataset, batch_size=batch_size)

    def training_step(state: TrainState, data: jnp.ndarray):
        def loss_fn(model_params: hk.Params, batch):
            loss = -jnp.mean(state.apply_fn(model_params, batch))
            return loss

        grads = jax.grad(loss_fn)(state.params, data)
        state = state.apply_gradients(grads=grads)

        return state, None

    def eval_fn(state: TrainState, data: jnp.ndarray):
        loss = -jnp.mean(state.apply_fn(state.params, data))
        return state, loss

    print(f"Train density estimator for {n_training_steps} episodes on proposal data.")
    for step in range(n_training_steps):
        rng_key, next_rng_key = jax.random.split(next_rng_key)
        batched_data = train_dataloader(rng_key)
        state, _ = jax.lax.scan(training_step, state, batched_data)

        if step % eval_frequency == 0:
            rng_key, next_rng_key = jax.random.split(next_rng_key)
            batched_data = eval_dataloader(rng_key)
            _, val_loss = jax.lax.scan(eval_fn, state, batched_data)
            print(f"STEP: {step}; validation loss: {jnp.mean(val_loss)}")

    return state


def run_rejection_sampling(
    rng_key,
    n_posterior_samples,
    model,
    prior,
    mixture_sampler,
    mixture_log_prob_fn,
    reference_data,
    support,
):
    """3. Carry out rejection sampling based on a mixture of proposal distribution.
    The proposal mixture consists of the prior distribution
    and the learned posterior model.
    :param rng_key: Random key
    :param n_posterior_samples: Number of desired posterior samples
    :param model: The joint model from which to draw the posterior samples
    :param prior: The prior of the model
    :param mixture_sampler: Samples from the pre-defined mixture distribution
    :param mixture_log_prob_fn: Evaluates the log prob of the mixture distribution
    :param reference_data: The reference observations on which to condition the posterior
    :param support: The param support in which to sample
    """
    print("Calibrating rejection sampling ...")
    rng_key, next_rng_key = jax.random.split(rng_key)
    # prior_samples = prior.sample(rng_key, (1000,))
    prior_samples = sample_within_support(
        rng_key, 1000, prior.sample, support, max_tries=1000
    )
    log_joint = jax.vmap(
        lambda params: log_density(
            model,
            model_args=(reference_data,),
            model_kwargs={},
            params={"params": params},
        )[0],
        0,
        0,
    )(prior_samples)
    # log_normalization = jnp.min(log_joint)

    log_normalization = -prior_samples.shape[0] + jax.scipy.special.logsumexp(log_joint)

    def find_m(x):
        rng_key, step, log_m = x
        rng_key, next_rng_key = jax.random.split(rng_key)

        # Create prior sample
        sample = sample_within_support(
            rng_key, 1, prior.sample, support, max_tries=1000
        )
        sample = sample[0]

        # Evaluate the log probabilities of the proposal distribution
        log_proposal = mixture_log_prob_fn(sample)

        # Evaluate the log probability of the joint
        log_joint = log_density(
            model,
            model_args=(reference_data,),
            model_kwargs={},
            params={"params": sample},
        )[0]
        log_ratio = (-log_normalization + log_joint - log_proposal).squeeze()
        return jax.lax.cond(
            jnp.greater(log_ratio, log_m),
            lambda: (next_rng_key, 0, jnp.log(1.01) + log_ratio),
            lambda: (next_rng_key, step + 1, log_m),
        )

    # find_m((next_rng_key, 0, 0.0))
    next_rng_key, _, log_m = jax.lax.while_loop(
        lambda x: x[1] != 10_000, find_m, (next_rng_key, 0, 0.0)
    )

    print(f"We found log(M) as {log_m}.")
    print("Running rejection sampling...")

    def rejection_sampling(x):
        """Running rejection sampling"""
        rng_key, posterior_samples, idx, steps = x
        rng_key1, rng_key2, next_rng_key = jax.random.split(rng_key, 3)
        sample = sample_within_support(
            rng_key1, 1, mixture_sampler, support, max_tries=1000
        )
        sample = sample[0]
        # sample = mixture_sampler(rng_key1, (1,))
        log_proposal = mixture_log_prob_fn(sample)
        log_joint = log_density(
            model,
            model_args=(reference_data,),
            model_kwargs={},
            params={"params": sample},
        )[0]
        uniform = jax.random.uniform(rng_key2, shape=(1,)).squeeze()
        log_ratio = (-log_normalization + log_joint - log_proposal - log_m).squeeze()
        hcb.call(
            callback_experiment_logger,
            {
                "step": steps,
                "metrics": {"accepted_samples": idx, "acceptance_ratio": idx / steps},
                "log_every_n_iterations": 1_000_000,
            },
            result_shape=None,
        )
        return jax.lax.cond(
            jnp.greater(log_ratio, jnp.log(uniform)),
            lambda: (
                next_rng_key,
                posterior_samples.at[idx].set(sample.squeeze()),
                idx + 1,
                steps + 1,
            ),
            lambda: (next_rng_key, posterior_samples, idx, steps + 1),
        )

    posterior_samples = jnp.empty((n_posterior_samples, prior_samples.shape[-1]))

    next_rng_key, posterior_samples, _, _ = jax.lax.while_loop(
        lambda x: x[2] < n_posterior_samples,
        rejection_sampling,
        (next_rng_key, posterior_samples, 0, 1),
    )
    return posterior_samples


def callback_experiment_logger(data: Dict):
    """Callback function to log the rejection sampling within a jitted function"""
    log_every_n_iterations = data.pop("log_every_n_iterations")
    if data["step"] % log_every_n_iterations == 0:
        step = data.pop("step")
        print(f"Step {step}")
        for key, val in data["metrics"].items():
            print(f"{key}: {val}")


def plot_samples(samples_dir: str, name: str, n_samples=10_000):
    """Pair plotting the posterior samples"""
    samples = np.load(os.path.join(samples_dir, f"{name}.npy"))
    n_samples = min(n_samples, samples.shape[0])
    rnd_idx = np.random.choice(np.arange(samples.shape[0]), n_samples, replace=False)
    df = pd.DataFrame(samples[rnd_idx])
    sns.pairplot(df)
    plt.savefig(
        os.path.join(samples_dir, f"{name}.png"),
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
