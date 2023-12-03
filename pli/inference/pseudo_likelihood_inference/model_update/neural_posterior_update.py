from typing import Dict, Tuple, Any

import jax
import jax.numpy as jnp
import haiku as hk
import optax

from pli.inference.pseudo_likelihood_inference.types import PLITrainState
from pli.models.density_estimators import nsf
from pli.models.density_estimators.types import DensityEstimator
from pli.utils.dataloaders import batched_data_from_perm_idx, permutation_indices


def build_neural_posterior_update(
    model: DensityEstimator,
    batch_size: int,
    training_steps: int,
    retrain_from_scratch: bool = True,
    update_method="m-projection",
    **_ignore,
):
    if update_method == "m-projection":

        def update_fn(params, likelihood, train_state: PLITrainState):
            return update(
                params,
                likelihood,
                train_state,
                model.log_prob,
                model.init,
                batch_size,
                training_steps,
                retrain_from_scratch,
            )

    else:
        raise ValueError(
            "Choose between the following update methods: `m-projection`"
        )
    return update_fn


def update(
    params,
    likelihood,
    train_state: PLITrainState,
    model_log_prob,
    model_init_fn,
    batch_size,
    training_steps,
    retrain_from_scratch,
) -> Tuple[PLITrainState, Dict]:
    weights = likelihood
    data = jnp.concatenate([params, weights[..., jnp.newaxis]], axis=-1)
    optimizer = train_state.optimizer

    def loss_fn(
        trainable_params: hk.Params, non_trainable_params: hk.Params, params, weights
    ):
        """Weighted maximum likelihood loss.
        :param trainable_params: Trainable parameters of the model.
        :param non_trainable_params: Non-trainable parameters of the model.
        :param params: Parameters of the reference distribution.
        :param weights: Weights of the reference distribution, i.e., the likelihood approximation.
        :returns: Weighted maximum likelihood loss.
        """
        return jnp.mean(
            -weights
            * model_log_prob(
                params, hk.data_structures.merge(trainable_params, non_trainable_params)
            )
        )

    @jax.jit
    def optimize(train_state: PLITrainState, data: Dict) -> Tuple[PLITrainState, Any]:
        """Single SGD update step.
        :param train_state: Current training state.
        :param data: Training data. Tuple of parameters and weights.
        """
        params = data["params"]
        weights = data["weights"]
        trainable_params, non_trainable_params = hk.data_structures.partition(
            nsf.filter_non_trainable_params, train_state.model_params
        )
        grads = jax.grad(loss_fn)(
            trainable_params, non_trainable_params, params, weights
        )

        updates, new_opt_state = optimizer.update(grads, train_state.model_opt_state)
        new_params = optax.apply_updates(trainable_params, updates)
        eval_loss = jax.lax.cond(
            train_state.model_update_steps % 100 == 0,
            lambda: loss_fn(new_params, non_trainable_params, params, weights),
            lambda: 0.0,
        )
        train_state = train_state.replace(
            model_params=hk.data_structures.merge(new_params, non_trainable_params),
            model_opt_state=new_opt_state,
            model_update_steps=train_state.model_update_steps + 1,
        )
        return (
            train_state,
            {"nsf loss": eval_loss, "nsf steps": train_state.model_update_steps},
        )

    def training_episode(train_state: PLITrainState, step: int):
        """Single training episode.
        :param train_state: Current training state.
        :param step: Current training step.
        """
        rng_key, next_rng_key = jax.random.split(train_state.rng_key)
        train_state = train_state.replace(rng_key=next_rng_key)

        # Batch training data
        perm_idx = permutation_indices(rng_key, data.shape[0])
        batched_params = batched_data_from_perm_idx(params, perm_idx, batch_size)
        batched_weights = batched_data_from_perm_idx(weights, perm_idx, batch_size)

        train_state, logs = jax.lax.scan(
            optimize, train_state, {"params": batched_params, "weights": batched_weights}
        )
        # hcb.id_tap(print_log, (logs, step, 100))
        logs.update(
            {
                "nsf episodes": jnp.repeat(
                    jnp.array(step * train_state.episode), batched_params.shape[0]
                )
            }
        )
        return train_state, logs

    def reset_state(state: PLITrainState):
        """Reset the training state.
        :param state: The current training state.
        :returns: Randomly initialize the parameters and the optimizer state.
        """
        rng_key, next_rng_key = jax.random.split(state.rng_key)
        model_params = model_init_fn(rng_key)
        trainable_params, _ = hk.data_structures.partition(
            nsf.filter_non_trainable_params, model_params
        )
        opt_states = optimizer.init(trainable_params)
        return state.replace(
            rng_key=next_rng_key, model_params=model_params, model_opt_state=opt_states
        )

    train_state = jax.lax.cond(
        retrain_from_scratch, reset_state, lambda state: state, train_state
    )
    train_state, logs = jax.lax.scan(
        training_episode, train_state, jnp.arange(training_steps)
    )
    logs = jax.tree_util.tree_map(lambda x: jnp.reshape(x, -1)[::100], logs)
    return train_state, logs
