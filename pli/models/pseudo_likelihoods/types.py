import dataclasses
from typing import Callable, Dict
from jax.random import PRNGKey
import jax.numpy as jnp


@dataclasses.dataclass
class PseudoLikelihood:
    init: Callable[[PRNGKey], Dict]
    sample: Callable[[PRNGKey, jnp.ndarray], jnp.ndarray]
    log_prob: Callable
    pdf: Callable
    log_prob_from_distance: Callable
    pdf_from_distance: Callable
    bandwidth: Callable[[], jnp.ndarray]
