import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)


class Plot:
    """Small plotting adapter used by the optional debug mode."""

    def __init__(self, image: np.ndarray, title: str = "Initial problem") -> None:
        aspect_ratio = image.shape[0] / image.shape[1]
        width = 8
        height = width * aspect_ratio
        figure = plt.figure(figsize=(width, height), frameon=False)
        axes = Axes(figure, (0.0, 0.0, 1.0, 0.9))
        axes.set_axis_off()
        figure.add_axes(axes)

        self._current_image = axes.imshow(image, aspect="auto", animated=True)
        self.show_fittest(image, title)

    def show_fittest(self, image: np.ndarray, title: str) -> None:
        """Display the current best solution."""
        plt.suptitle(title, fontsize=20)
        self._current_image.set_data(image)
        plt.draw()
        plt.pause(0.05)
