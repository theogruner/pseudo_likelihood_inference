import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from numpyro.distributions import LogNormal


@jax.tree_util.register_pytree_node_class
class GaussianDistribution:
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov

    def sample(self, rng_key: jnp.ndarray, sample_shape=()):
        """Sampling from the multivariate distribution"""
        return jax.random.multivariate_normal(
            rng_key,
            self.mean,
            self.cov,
            sample_shape,
        )

    def log_prob(self, x: jnp.ndarray):
        """Evaluate the log-prob of the multivariate Gaussian distribution"""
        return jax.scipy.stats.multivariate_normal.logpdf(x, self.mean, self.cov)

    def tree_flatten(self):
        children = (self.mean, self.cov)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
class LogNormalDistribution:
    """Multivariate LogNormal, i.e., the log-prob is summed over the last dimension compared
    to the numpyro implementation."""

    def __init__(self, mean, var):
        self.mean = mean
        self.var = var
        self.dist = LogNormal(mean, var)

    def sample(self, rng_key: jnp.ndarray, sample_shape=()):
        """Sampling from a log-normal distribution"""
        return self.dist.sample(rng_key, sample_shape)

    def log_prob(self, x: jnp.ndarray):
        """Evaluate the log-prob of the log-normal distribution"""
        return self.dist.log_prob(x).sum(axis=-1)

    def tree_flatten(self):
        children = (self.mean, self.var)
        aux_data = self.dist
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
class GaussianMixtureDistribution:
    def __init__(self, mixture_coeffs, means, covs):
        self.mixture_coeffs = mixture_coeffs
        self.means = means
        self.covs = covs

    def sample(self, rng_key: jnp.ndarray, sample_shape=()):
        """Sampling from the Gaussian mixture distribution"""
        rng_key1, rng_key2 = jax.random.split(rng_key)
        components_idx = jax.random.categorical(
            rng_key1, logits=self.mixture_coeffs, shape=sample_shape
        )
        return jax.random.multivariate_normal(
            rng_key2, self.means[components_idx], self.covs[components_idx]
        )

    def log_prob(self, x: jnp.ndarray):
        """Evaluate the log-prob of the Gaussian mixture distribution"""
        normal_pdfs = jax.vmap(
            lambda mean, cov: jax.scipy.stats.multivariate_normal.logpdf(x, mean, cov)
        )(self.means, self.covs)
        return logsumexp(jnp.log(self.mixture_coeffs) + normal_pdfs, axis=-1)

    def tree_flatten(self):
        children = (self.mixture_coeffs, self.means, self.covs)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
