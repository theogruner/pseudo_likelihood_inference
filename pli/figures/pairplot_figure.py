from typing import List, Optional, Dict

import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

from pli.figures.figures import Figure


class PairplotFigure(Figure):
    name = "PairplotFigure"

    def __init__(
        self,
        param_names: List[str],
        param_support: Optional[Dict] = None,
        n_samples=1000,
        n_plotting_params: Optional[int] = None,
        **_ignore
    ):
        super().__init__()
        self.param_names = param_names
        self._fig = None
        self.param_support = param_support
        self.n_samples = n_samples
        if param_support is not None:
            self.param_support = np.array(
                [
                    [param_support[key][0] for key in self.param_names],
                    [param_support[key][1] for key in self.param_names],
                ]
            )
        else:
            self.param_support = param_support

        self._n_plotting_params = n_plotting_params
        if n_plotting_params is None or n_plotting_params > len(param_names):
            self._n_plotting_params = len(param_names)

    @property
    def n_plotting_params(self):
        return self._n_plotting_params

    @n_plotting_params.setter
    def n_plotting_params(self, value):
        if value > len(self.param_names):
            self._n_plotting_params = len(self.param_names)
        else:
            self._n_plotting_params = value

    def update(self, data):
        if isinstance(self._fig, sns.PairGrid):
            # Otherwise, multiple plotting instances are opened.
            plt.close(self._fig.fig)
        dfs = []
        if "posterior_params" in data and data["posterior_params"] is not None:
            posterior_data = data["posterior_params"][:, :self.n_plotting_params]
            if posterior_data.shape[0] > self.n_samples:
                posterior_data = posterior_data[: self.n_samples]
            df = pd.DataFrame(
                data=posterior_data, columns=self.param_names[:self.n_plotting_params]
            )
            df["label"] = "Posterior"
            dfs.append(df)
        if "reference_params" in data:
            ref_data = data["reference_params"][0, :self.n_plotting_params]
            if ref_data.ndim == 1:
                ref_data = np.atleast_2d(ref_data)
            if ref_data.shape[0] > self.n_samples:
                ref_data = ref_data[: self.n_samples]
            df = pd.DataFrame(
                data=ref_data, columns=self.param_names[:self.n_plotting_params]
            )
            df["label"] = "Reference"
            dfs.append(df)
        df = pd.DataFrame(
            data=data["params"][:, :self.n_plotting_params],
            columns=self.param_names[:self.n_plotting_params]
        )
        df["label"] = "Predictions"
        dfs.append(df)

        dfs = pd.concat(dfs, ignore_index=True)
        self._fig = sns.pairplot(dfs, hue="label", diag_kws={"common_norm": False})
        self._set_axis_limits()

    def _set_axis_limits(self):
        if self.param_support is not None:
            # Check non-diagonal scatter plots
            for i in range(self.n_plotting_params):
                for j in range(self.n_plotting_params):
                    self._fig.axes[i, j].set_xlim(*self.param_support[:, j])
                    if i != j:
                        self._fig.axes[i, j].set_ylim(*self.param_support[:, i])

    @property
    def fig(self):
        return self._fig.fig

    def show(self):
        self._fig.fig.show()
