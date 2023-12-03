from typing import Callable, Tuple, Optional, Dict

import jax.numpy as jnp
import optax

from pli.models.density_estimators import nsf
from .gaussian_model_update import build_gaussian_update
from .neural_posterior_update import build_neural_posterior_update
from ..types import PLITrainState


def build_model_update(
    model_name: str, model, **kwargs
) -> Callable[[jnp.ndarray, jnp.ndarray, PLITrainState], Tuple[PLITrainState, Dict]]:
    """Depending on the choice of the model, a different update method is selected."""
    if model_name in ["NSF", "CNF"]:
        update_method = build_neural_posterior_update(model, **kwargs)
    elif model_name == "Gaussian":
        update_method = build_gaussian_update(**kwargs)
    else:
        raise ValueError(
            "Choose between one of the following density estimators:"
            "`Gaussian`, `GMM`, `NSF`,`PMCSampler`, `SMCSampler`"
        )
    return update_method


def build_optimizer(
        name, lr=None, **_ignore
) -> Tuple[Optional[optax.GradientTransformation], Optional[Callable]]:
    """Build an optimizer for all models that are trained by gradient descent.
    In this case, it's only the 'NSF' model"""
    optimizer = None
    filter_non_trainable_params_fn = None
    if lr is not None:
        optimizer = optax.adam(lr)
        if name == "NSF":
            filter_non_trainable_params_fn = nsf.filter_non_trainable_params
    return optimizer, filter_non_trainable_params_fn
