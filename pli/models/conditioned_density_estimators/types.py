import dataclasses
from typing import Callable, Tuple
from jax.random import PRNGKey
from jaxtyping import PyTree, Array


@dataclasses.dataclass
class ConditionedDensityEstimator:
    """Conditioned Density Estimator"""
    init: Callable[[PRNGKey], PyTree]
    sample: Callable[[PRNGKey, Tuple, PyTree], Array]
    log_prob: Callable[[Array, Array, PyTree], Array]
    pdf: Callable[[Array, PyTree], Array]
