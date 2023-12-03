import dataclasses
from typing import Callable, Tuple
from jax.random import PRNGKey
from jaxtyping import PyTree, Array


@dataclasses.dataclass
class ParticleSampler:
    """Particle sampler"""
    init: Callable[[PRNGKey, int], PyTree]
    sample: Callable[[PRNGKey, Tuple, PyTree], Array]
    log_prob: Callable[[Array, PyTree], Array]
    pdf: Callable[[Array, PyTree], Array]
