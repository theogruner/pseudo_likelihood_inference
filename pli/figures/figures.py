from abc import ABC, abstractmethod


class Figure(ABC):
    _fig = None

    @abstractmethod
    def update(self, data):
        """Update the figure with new data."""

    @property
    def fig(self):
        return self._fig

    def show(self):
        """Show the figure."""
        self._fig.show()
