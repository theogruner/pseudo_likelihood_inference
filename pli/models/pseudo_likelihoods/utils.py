from typing import Callable

from pli.models.pseudo_likelihoods.exponential_kernel import build_exponential_kernel
from pli.models.pseudo_likelihoods.types import PseudoLikelihood
from pli.models.pseudo_likelihoods.uniform_kernel import build_uniform_kernel


def build_pseudo_likelihood(
    name, simulator: Callable, distance_fn, n_samples_per_param, **kwargs
) -> PseudoLikelihood:
    if name == "ExponentialKernel":
        return build_exponential_kernel(simulator, distance_fn, n_samples_per_param, **kwargs)

    elif name == "UniformKernel":
        return build_uniform_kernel(simulator, distance_fn, n_samples_per_param, **kwargs)
    else:
        raise ValueError(
            "The pseudo-likelihood is either an `ExponentialKernel` or a `UniformKernel`, "
            f"but {name} was given."
        )
