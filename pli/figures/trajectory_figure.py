from typing import Dict

import numpy as np
from matplotlib import pyplot as plt

from pli.figures.figures import Figure


class TrajectoryFigure(Figure):
    name = "TrajectoryFigure"

    def __init__(self, observation_dim, alpha=0.02, lw=2.0, **_ignore):
        """Figure, plotting the trajectories of the observations and targets.

        param observation_dim:
        param alpha: Opacity/alpha value at 1000 samples.
        Based on that, an adaptive alpha value is calculated.
        """
        super().__init__()
        self._alpha = alpha
        self._lw = lw
        self._nrows, self._ncols = observation_dim, 1
        self._fig, self._axs = plt.subplots(nrows=self._nrows, ncols=self._ncols)
        if not hasattr(self._axs, "__iter__"):
            self._axs = [self._axs]

    def update(self, data: Dict):
        observations = data["observations"]
        targets = data["targets"] if "targets" in data else None
        alpha = self._alpha  # self._alpha ** (observations.shape[0] / 1000)
        len_episode = observations.shape[2]
        predict_time = np.tile(np.arange(0.0, len_episode), (observations.shape[0], 1))
        alpha = self._alpha ** (targets.shape[0] / 1000)
        if targets is not None:
            target_time = np.tile(np.arange(0.0, len_episode), (targets.shape[0], 1))
        for i in range(self._nrows):
            ax = self._axs[i]
            ax.cla()
            if targets is not None:
                ax.plot(
                    target_time.transpose(),
                    targets[..., i].transpose(),
                    color="tab:blue",
                    alpha=alpha,
                    lw=self._lw,
                )
            ax.plot(
                predict_time.transpose(),
                observations[0, ..., i].transpose(),
                color="tab:orange",
                alpha=alpha,
                lw=self._lw,
            )
            if i != self._nrows - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("t")
