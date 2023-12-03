from typing import Optional

from jaxtyping import Array

from .conditioned_density_estimators.conditioned_nsf import build_conditioned_nsf
from .density_estimators.gaussian import build_gaussian
from .density_estimators.nsf import build_nsf
from .density_estimators.types import DensityEstimator


def build_density_estimator(
    name: str,
    param_support: Optional[Array] = None,
    prior_sample: Optional[Array] = None,
    n_dim: Optional[int] = None,
    **kwargs
) -> DensityEstimator:
    """Build a density estimator"""
    if name == "Gaussian":
        density_estimator = build_gaussian(param_support, n_dim=n_dim, **kwargs)
    elif name == "NSF":
        density_estimator = build_nsf(param_support, prior_sample, **kwargs)
    else:
        raise ValueError(
            "Choose between one of the following density estimators:"
            "`Gaussian`, `GMM`, `NSF`, `PMCSampler`, `SMCSampler`"
        )
    return density_estimator


def build_conditioned_density_estimator(
    name: str,
    param_support: Optional[Array] = None,
    prior_sample: Array = None,
    simulation: Array = None,
    **kwargs
):
    """Build a conditional density estimator"""
    if name == "ConditionedNSF":
        density_estimator = build_conditioned_nsf(
            param_support, prior_sample, simulation, **kwargs
        )
    else:
        raise ValueError(
            "Choose between one of the following conditioned density estimators:"
            "\n`ConditionedNSF`"
        )
    return density_estimator
