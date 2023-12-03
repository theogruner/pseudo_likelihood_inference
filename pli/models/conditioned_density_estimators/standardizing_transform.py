import jax.numpy as jnp
import haiku as hk


class StandardizingTransform(hk.Module):
    """Standardizing Transform
    """
    def __init__(self, t_mean, t_std, name=None):
        """
        Fixed standardized transformation that shifts the input along
        't_mean' and scales by 1 / 't_std'.
        :param t_mean: Mean of the input data
        :param t_std: Standard deviation of the input data
        """
        super().__init__(name)
        self.t_mean = t_mean
        self.t_std = t_std

    def __call__(self, x):
        t_mean_init = hk.initializers.Constant(self.t_mean)
        t_std_init = hk.initializers.Constant(self.t_std)
        t_mean = hk.get_parameter(
            "t_mean", shape=[*self.t_mean.shape], init=t_mean_init
        )[jnp.newaxis]
        t_std = hk.get_parameter("t_std", shape=[*self.t_std.shape], init=t_std_init)[jnp.newaxis]
        return (x - t_mean) / t_std
