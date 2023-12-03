import dataclasses
from typing import Callable, Union

from pli.models.conditioned_density_estimators.types import ConditionedDensityEstimator
from pli.models.density_estimators.types import DensityEstimator


@dataclasses.dataclass
class SBIExperiment:
    init: Callable
    step: Callable
    evaluate: Callable
    model: Union[ConditionedDensityEstimator, DensityEstimator]
