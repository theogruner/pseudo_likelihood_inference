from typing import Sequence

import distrax
import haiku as hk
import jax.numpy as jnp
import numpy as np
from jaxtyping import PyTree

from .types import DensityEstimator


def build_nsf(
    param_support,
    prior_sample,
    num_layers=5,
    hidden_size=50,
    mlp_num_layers=2,
    num_bins=10,
    **_ignore,
):
    """Build a neural spline flow [1]
    [1] Durkan, C., Bekasov, A., Murray, I., & Papamakarios, G. (2019). Neural spline flows. """
    event_shape = (param_support.shape[-1],)
    t_mean = jnp.mean(prior_sample, axis=0)
    t_std = jnp.std(prior_sample, axis=0)
    t_std = jnp.where(t_std < 1e-14, 1e-14 * jnp.ones_like(t_std), t_std)

    forward_log_prob = forward_fn_log_prob(
        event_shape,
        num_layers,
        hidden_size,
        mlp_num_layers,
        num_bins,
        t_mean_init=t_mean,
        t_std_init=t_std,
    )
    forward_sample = forward_fn_sample(
        event_shape,
        num_layers,
        hidden_size,
        mlp_num_layers,
        num_bins,
        t_mean_init=t_mean,
        t_std_init=t_std,
    )

    def log_prob(x, model_params):
        return forward_log_prob.apply(model_params, x=x)

    def _init_flow(rng_key: jnp.ndarray) -> PyTree:
        model_params = forward_log_prob.init(
            rng_key, jnp.ones((1, param_support.shape[-1]))
        )
        return model_params

    return DensityEstimator(
        init=_init_flow,
        sample=lambda rng_key, sample_shape, model_params: forward_sample.apply(
            model_params, sample_shape=sample_shape, rng_key=rng_key
        ),
        log_prob=log_prob,
        pdf=lambda x, model_params: jnp.exp(log_prob(x, model_params)),
    )


def make_conditioner(
    event_shape: Sequence[int], hidden_sizes: Sequence[int], num_bijector_params: int
) -> hk.Sequential:
    """Creates an MLP conditioner for each layer of the flow."""
    return hk.Sequential(
        [
            hk.Flatten(preserve_dims=-1),
            hk.nets.MLP(hidden_sizes, activate_final=True),
            # We initialize this linear layer to zero so that the flow is initialized
            # to the identity function.
            hk.Linear(
                np.prod(event_shape) * num_bijector_params,
                w_init=jnp.zeros,
                b_init=jnp.zeros,
            ),
            hk.Reshape(tuple(event_shape) + (num_bijector_params,), preserve_dims=-1),
        ]
    )


def make_flow_model(
    event_shape,
    num_layers: int,
    hidden_sizes: Sequence[int],
    num_bins: int,
    t_mean_init,
    t_std_init,
) -> distrax.Transformed:
    """Creates the flow model."""
    # Alternating binary mask.
    mask = jnp.arange(0, np.prod(event_shape)) % 2
    mask = jnp.reshape(mask, event_shape)
    mask = mask.astype(bool)

    num_bijector_params = 3 * num_bins + 1

    def bijector_fn(params: jnp.ndarray):
        return distrax.RationalQuadraticSpline(
            params,
            range_min=-3.0,
            range_max=3.0,
        )

    # Number of parameters for the rational-quadratic spline:
    # - `num_bins` bin widths
    # - `num_bins` bin heights
    # - `num_bins + 1` knot slopes
    # for a total of `3 * num_bins + 1` parameters.

    # Z-scoring input
    t_mean_init = hk.initializers.Constant(t_mean_init)
    t_std_init = hk.initializers.Constant(t_std_init)
    t_mean = hk.get_parameter("t_mean", shape=[*event_shape], init=t_mean_init)
    t_std = hk.get_parameter("t_std", shape=[*event_shape], init=t_std_init)
    init_layer = distrax.LowerUpperTriangularAffine(
        matrix=jnp.diag(1 / t_std), bias=-t_mean / t_std
    )
    layers = []
    for _ in range(num_layers):
        layer = distrax.MaskedCoupling(
            mask=mask,
            bijector=bijector_fn,
            conditioner=make_conditioner(
                event_shape=event_shape,
                hidden_sizes=hidden_sizes,
                num_bijector_params=num_bijector_params,
            ),
        )
        layers.append(layer)
        # Flip the mask after each layer.
        mask = jnp.logical_not(mask)
    layers.append(init_layer)

    # We invert the flow so that the `forward` method is called with `log_prob`.
    flow = distrax.Inverse(distrax.Chain(layers))
    base_distribution = distrax.Independent(
        # distrax.Uniform(low=param_support[0], high=param_support[1]),
        # distrax.Uniform(low=jnp.zeros(event_shape), high=jnp.ones(event_shape)),
        # reinterpreted_batch_ndims=len(event_shape),
        distrax.Normal(loc=jnp.zeros(event_shape), scale=jnp.ones(event_shape)),
        reinterpreted_batch_ndims=len(event_shape),
    )

    return distrax.Transformed(base_distribution, flow)


def forward_fn_log_prob(
    event_shape,
    num_layers,
    hidden_size,
    mlp_num_layers,
    num_bins,
    t_mean_init,
    t_std_init,
):
    @hk.without_apply_rng
    @hk.transform
    def log_prob(x: jnp.ndarray) -> jnp.ndarray:
        model = make_flow_model(
            event_shape,
            num_layers=num_layers,
            hidden_sizes=[hidden_size] * mlp_num_layers,
            num_bins=num_bins,
            t_mean_init=t_mean_init,
            t_std_init=t_std_init,
        )
        return model.log_prob(x)

    return log_prob


def forward_fn_sample(
    event_shape,
    num_layers,
    hidden_size,
    mlp_num_layers,
    num_bins,
    t_mean_init,
    t_std_init,
):
    @hk.without_apply_rng
    @hk.transform
    def sample(rng_key, sample_shape) -> jnp.ndarray:
        model = make_flow_model(
            event_shape,
            num_layers=num_layers,
            hidden_sizes=[hidden_size] * mlp_num_layers,
            num_bins=num_bins,
            t_mean_init=t_mean_init,
            t_std_init=t_std_init,
        )
        return model.sample(seed=rng_key, sample_shape=sample_shape)

    return sample


def filter_non_trainable_params(m, n, p):
    return n not in ('t_mean', 't_std')
