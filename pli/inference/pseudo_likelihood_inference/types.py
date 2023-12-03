from typing import Any, Optional

import jax.numpy as jnp
import optax
from flax import struct


@struct.dataclass
class PLITrainState:
    optimizer: optax.GradientTransformation = struct.field(pytree_node=False)
    rng_key: jnp.ndarray = None
    likelihood_params: Any = None
    model_params: Any = None
    model_opt_state: Optional[optax.OptState] = None

    log_eta: jnp.ndarray = None
    nu: float = None
    epsilon: float = None

    # Logging
    n_simulations: int = 0
    episode: int = 0
    model_update_steps: int = 0
