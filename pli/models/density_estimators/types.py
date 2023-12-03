import dataclasses
from typing import Callable, Tuple
from jax.random import PRNGKey
from jaxtyping import PyTree, Array


@dataclasses.dataclass
class DensityEstimator:
    init: Callable[[PRNGKey], PyTree]
    sample: Callable[[PRNGKey, Tuple, PyTree], Array]
    log_prob: Callable[[Array, PyTree], Array]
    pdf: Callable[[Array, PyTree], Array]
