import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
class UniformDistribution:
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def sample(self, rng_key: jnp.ndarray, sample_shape=()):
        """Sampling from the uniform distribution"""
        return jax.random.uniform(
            rng_key,
            (*sample_shape, self.min_val.size),
            minval=self.min_val,
            maxval=self.max_val,
        )

    def log_prob(self, x: jnp.ndarray):
        """Evaluate the log-prob of the uniform distribution"""
        d = self.max_val - self.min_val
        return jnp.where(
            jnp.logical_or(jnp.all(x < self.min_val), jnp.all(x > self.max_val)),
            jnp.log(jnp.zeros(x.shape[:-1])),
            jnp.sum(jnp.log(1 / d)) * jnp.ones(x.shape[:-1])
        )

    def tree_flatten(self):
        """Flatten the pytree"""
        children = (self.min_val, self.max_val)
        aux_data = ()
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten the pytree"""
        return cls(*children)
