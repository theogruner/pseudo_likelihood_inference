from matplotlib import pyplot as plt

from ..figures import Figure


class GaussianFigure(Figure):
    """Plot of the observation space of the simple-likelihood,
    complex-posterior (SLCP) environment.
    It consists of four samples from a 2D-Gaussian.
    """
    name = "GaussianFigure"

    def __init__(self):
        super().__init__()
        self._fig, self._ax = plt.subplots()
        self.x_lim = (-4., 4.)
        self.y_lim = (-4., 8.)

    def update(self, data):
        observations = data["observations"]
        observations = observations[:, 0, ...].reshape((observations.shape[0], 1, 4, 2))
        targets = data["targets"] if "targets" in data else None

        self._ax.cla()
        if targets is not None:
            targets = targets.reshape((targets.shape[0], 1, 4, 2))
            for target in targets:
                self._ax.scatter(target[0, :, 0], target[0, :, 1], color="tab:blue", alpha=0.8)
        for observation in observations:
            self._ax.scatter(
                observation[0, :, 0], observation[0, :, 1], color="tab:orange", alpha=0.5
            )
        self._ax.set_xlim(self.x_lim)
        self._ax.set_ylim(self.y_lim)
