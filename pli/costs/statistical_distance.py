import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array
from ott.core import linear_problems, sinkhorn
from ott.geometry import pointcloud


def mse(x, y):
    """
    works only if x.shape[0] == y.shape[0]
    """
    flattend_dim = np.prod(x.shape[1:])
    n = x.shape[0]
    x = x.reshape((x.shape[0], flattend_dim))
    y = y.reshape((y.shape[0], flattend_dim))

    squared_euclidean = jnp.sum((x - y) ** 2)

    return squared_euclidean / n


def wasserstein(x: Array, y: Array) -> Array:
    """
    Wasserstein distance on trajectory data. The distance is computed with the jax ott package.
    :param x: Empirical data of Dim[N, T, D]
    :param y: Empirical data of Dim[M, T, D]
    """
    flattend_dim = np.prod(x.shape[1:])
    x = x.reshape((x.shape[0], flattend_dim))
    y = y.reshape((y.shape[0], flattend_dim))

    geom = pointcloud.PointCloud(x, y)
    ot_prob = linear_problems.LinearProblem(geom)
    solver = sinkhorn.Sinkhorn()
    out = solver(ot_prob)
    return out.reg_ot_cost  # / (n * m)  # , out


def memory_efficient_mmd(x: Array, y: Array) -> Array:
    """
    Maximum-mean discrepancy on batched data.

    :param x: Empirical data of Dim[N, T, D]
    :param y: Empirical data of Dim[M, T, D]
    """
    flattend_dim = np.prod(x.shape[1:])
    n = x.shape[0]
    m = y.shape[0]
    x = x.reshape((x.shape[0], flattend_dim))
    y = y.reshape((y.shape[0], flattend_dim))

    # Calculate kxx, kxy, and kyy
    kxx = _memory_efficient_mmd_kernel(x, x)
    i, j = jnp.diag_indices(kxx.shape[-1])
    kxx = kxx.at[..., i, j].set(0.0)

    kyy = _memory_efficient_mmd_kernel(y, y)
    i, j = jnp.diag_indices(kyy.shape[-1])
    kyy = kyy.at[..., i, j].set(0.0)

    kxy = _memory_efficient_mmd_kernel(x, y)
    return (
        (1 / (n * (n - 1))) * jnp.sum(kxx)
        - (2.0 / (n * m)) * jnp.sum(kxy)
        + (1 / (m * (m - 1))) * jnp.sum(kyy)
    )


def _memory_efficient_mmd_kernel(x, y) -> Array:
    """Calculate the kernel distance. For now, only the rbf kernel is considered.

    :param x: Empirical data of Dim[N, D]
    :param y: Empirical data of Dim[M, D]
    """
    # We first calculate the squared euclidean distance efficiently in terms of memory
    squared_distance = jnp.zeros((x.shape[0], y.shape[0]))

    def update_squared_distance(sd, data):
        """
        :param sd: Squared distance matrix
        :param data: PyTree containing 'x' and 'y' data with Dim[M] and Dim[N]
        """

        def squared_distance_fn(x, y):
            return (x - y) ** 2

        mapx1 = jax.vmap(squared_distance_fn, (None, 0), 0)
        mapx2 = jax.vmap(mapx1, (0, None), 0)
        return sd + mapx2(data["x"], data["y"]), None

    squared_distance, _ = jax.lax.scan(
        update_squared_distance,
        squared_distance,
        {"x": x.transpose(), "y": y.transpose()},
    )

    bandwidths = jnp.array(
        [1.0, 10.0, 20.0, 40.0, 80.0, 100.0, 130.0, 200.0, 400.0, 800.0, 1000.0]
    )

    kernelized_distance_matrix = jnp.zeros((x.shape[0], y.shape[0]))

    def rbf_kernel(kernelized_dist, bw):
        return kernelized_dist + jnp.exp(-0.5 / bw * squared_distance), None

    kernelized_distance_matrix, _ = jax.lax.scan(rbf_kernel, kernelized_distance_matrix, bandwidths)

    return kernelized_distance_matrix
