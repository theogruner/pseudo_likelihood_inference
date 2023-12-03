import dataclasses
from typing import Callable


@dataclasses.dataclass
class PerturbationKernel:
    """Perturbation kernel"""
    init: Callable
    forward: Callable
    log_prob: Callable
    sample: Callable
    update: Callable
