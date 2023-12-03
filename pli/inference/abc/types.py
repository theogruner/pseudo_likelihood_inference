from typing import Any

import jax.numpy as jnp
from flax import struct


@struct.dataclass
class ABCTrainState:
    rng_key: jnp.ndarray = None
    likelihood_params: Any = None
    model_params: Any = None

    previous_likelihood: jnp.ndarray = None
    current_distance: jnp.ndarray = jnp.array([])
    best_distance: jnp.ndarray = jnp.array([])

    # Logging
    n_simulations: int = 0
    episode: int = 0
    model_update_steps: int = 0
