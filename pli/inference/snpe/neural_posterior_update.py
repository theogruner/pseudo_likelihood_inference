from typing import Dict, Tuple, Any

import jax
import jax.numpy as jnp
import haiku as hk
import optax

from pli.inference.snpe.types import SNPETrainState
from pli.utils.dataloaders import permutation_indices, batched_data_from_perm_idx
from pli.models.conditioned_density_estimators import conditioned_nsf
from pli.models.conditioned_density_estimators.types import ConditionedDensityEstimator


def build_neural_posterior_update(
        name,
        prior,
        model: ConditionedDensityEstimator,
        batch_size,
        training_steps,
        n_atoms,
        retrain_from_scratch=True,
        **_ignore,
):
    if name == "snpe":

        def update_method(
            params, simulations, train_state: SNPETrainState, prior_proposal: bool
        ):
            return snpe_update(
                prior_proposal,
                params,
                simulations,
                train_state,
                model.log_prob,
                model.init,
                prior.log_prob,
                batch_size,
                training_steps,
                n_atoms,
                retrain_from_scratch,
            )

    else:
        raise ValueError("Choose between the following update methods:  'snpe`")
    return update_method


def snpe_update(
    prior_proposal,
    params,
    simulations,
    train_state: SNPETrainState,
    model_log_prob,
    model_init_fn,
    prior_log_prob,
    batch_size,
    train_steps,
    n_atoms,
    retrain_from_scratch,
    n_evals=100,
):
    """Update for a conditional density estimator with atomic proposals [1]."""

    log_freq = int((params.shape[0] / batch_size) * train_steps / n_evals)
    optimizer = train_state.optimizer

    def loss_fn(trainable_params, non_trainable_params, rng_key, params, data_summary):
        """Maximizing the log-proposal posterior with atomic proposals.
        :param model_params:
        :param params:
        :param data_summary:
        """
        # Create atomic proposals
        atomic_probs = (
            jnp.ones((batch_size, batch_size))
            * (1 - jnp.eye(batch_size))
            / (batch_size - 1)
        )
        contrastive_thetas = jax.vmap(
            jax.random.choice, (None, None, None, None, 0), 0
        )(
            rng_key,
            params,
            (n_atoms - 1,),
            False,
            atomic_probs,
        )
        atomic_params = jnp.concatenate(
            (params[:, jnp.newaxis, :], contrastive_thetas), axis=1
        ).reshape((n_atoms * batch_size, -1))
        repeated_data_summary = jnp.repeat(data_summary, n_atoms, axis=0)

        posterior_lp = model_log_prob(
            atomic_params,
            repeated_data_summary,
            hk.data_structures.merge(trainable_params, non_trainable_params),
        )
        posterior_lp = jnp.reshape(posterior_lp, (batch_size, n_atoms))
        prior_lp = prior_log_prob(atomic_params)
        prior_lp = jnp.reshape(prior_lp, (batch_size, n_atoms))
        proposal_lp = posterior_lp - prior_lp
        proposal_lp = proposal_lp[:, 0] - jax.scipy.special.logsumexp(
            proposal_lp, axis=-1
        )
        return -jnp.mean(proposal_lp)

    def first_round_loss(
        trainable_params, non_trainable_params, rng_key, params, data_summary
    ):
        """Maximum-likelihood loss on the neural posterior."""
        return -jnp.mean(
            model_log_prob(
                params,
                data_summary,
                hk.data_structures.merge(trainable_params, non_trainable_params),
            )
        )

    @jax.jit
    def optimize(train_state: SNPETrainState, data: Dict) -> Tuple[SNPETrainState, Any]:
        """Single SGD update step."""
        params = data["params"]
        sims = data["simulations"]
        rng_key, next_rng_key = jax.random.split(train_state.rng_key)
        trainable_params, non_trainable_params = hk.data_structures.partition(
            conditioned_nsf.filter_non_trainable_params, train_state.model_params
        )

        _, grads = jax.lax.cond(
            prior_proposal,
            lambda: jax.value_and_grad(first_round_loss)(
                trainable_params,
                non_trainable_params,
                rng_key,
                params,
                sims,  # data_summary
            ),
            lambda: jax.value_and_grad(loss_fn)(
                trainable_params,
                non_trainable_params,
                rng_key,
                params,
                sims,  # data_summary
            ),
        )

        updates, new_opt_state = optimizer.update(grads, train_state.model_opt_state)
        new_params = optax.apply_updates(trainable_params, updates)

        eval_loss = jax.lax.cond(
            train_state.model_update_steps % 100 == 0,
            lambda: first_round_loss(
                new_params, non_trainable_params, None, params, sims
            ),  # data_summary),
            lambda: 0.0,
        )
        train_state = train_state.replace(
            rng_key=next_rng_key,
            model_params=hk.data_structures.merge(new_params, non_trainable_params),
            model_opt_state=new_opt_state,
            model_update_steps=train_state.model_update_steps + 1,
        )
        return (
            train_state,
            {"snpe loss": eval_loss, "snpe steps": train_state.model_update_steps},
        )

    def training_episode(train_state: SNPETrainState, step: int):
        rng_key, next_rng_key = jax.random.split(train_state.rng_key)
        train_state = train_state.replace(rng_key=next_rng_key)

        # Batch training data
        perm_idx = permutation_indices(rng_key, params.shape[0])
        batched_params = batched_data_from_perm_idx(params, perm_idx, batch_size)
        batched_simulations = batched_data_from_perm_idx(
            simulations, perm_idx, batch_size
        )
        train_state, logs = jax.lax.scan(
            optimize,
            train_state,
            {"params": batched_params, "simulations": batched_simulations},
        )
        logs.update(
            {
                "snpe episodes": jnp.repeat(
                    jnp.array(step * train_state.episode), batched_params.shape[0]
                )
            }
        )
        return train_state, logs

    def reset_state(state: SNPETrainState):
        """Reset the parameters of the neural posterior and the optimizer state."""
        rng_key, next_rng_key = jax.random.split(state.rng_key)
        model_params = model_init_fn(rng_key, simulations[:1])
        trainable_params, _ = hk.data_structures.partition(
            conditioned_nsf.filter_non_trainable_params, model_params
        )
        opt_states = optimizer.init(trainable_params)
        return state.replace(
            rng_key=next_rng_key, model_params=model_params, model_opt_state=opt_states
        )

    train_state = jax.lax.cond(
        retrain_from_scratch,
        reset_state,
        lambda state: state,
        train_state
    )
    train_state, logs = jax.lax.scan(
        training_episode, train_state, jnp.arange(train_steps)
    )
    logs = jax.tree_util.tree_map(lambda x: jnp.reshape(x, -1)[::log_freq], logs)
    return train_state, logs
