from typing import Any, Optional
import jax.numpy as jnp
import optax
from flax import struct


@struct.dataclass
class SNPETrainState:
    optimizer: optax.GradientTransformation = struct.field(pytree_node=False)
    rng_key: jnp.ndarray = None
    model_params: Any = None
    model_opt_state: Optional[optax.OptState] = None

    n_simulations: int = 0
    episode: int = 0
    model_update_steps: int = 0
