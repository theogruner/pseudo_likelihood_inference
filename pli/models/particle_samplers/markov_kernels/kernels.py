from .independant_kernel import (
    build_multivariate_gaussian_kernel,
    build_gaussian_kernel,
    build_gmm_kernel,
)
from .conditional_kernel import (
    build_multivariate_conditional_gaussian_kernel,
    build_conditional_gaussian_kernel,
)


def build_markov_kernel(name: str):
    """Build a markov kernel"""
    if name == "MultivariateGaussianKernel":
        return build_multivariate_gaussian_kernel()
    if name == "GaussianKernel":
        return build_gaussian_kernel()
    if name == "GMMKernel":
        return build_gmm_kernel()
    if name == "ConditionalMultivariateGaussianKernel":
        return build_multivariate_conditional_gaussian_kernel()
    if name == "ConditionalGaussianKernel":
        return build_conditional_gaussian_kernel()

    raise ValueError("Please choose between one of the following Markov kernels:"
                     "\n'MultivariateGaussianKernel', 'GaussianKernel', 'GMMKernel', "
                     "ConditionalMultivariateGaussianKernel, 'ConditionalGaussianKernel'"
                     f"Your current choice is: '{name}'")
