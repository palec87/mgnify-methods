import matplotlib
matplotlib.use("Agg")

import numpy as np

from mgnify_methods.utils.plot import plot_mean_ci, plot_mean_std


def test_plot_mean_ci_smoke():
    x = np.array([1, 2, 3])
    mean_y = np.array([1.0, 2.0, 3.0])
    ci_lower = np.array([0.5, 1.5, 2.5])
    ci_upper = np.array([1.5, 2.5, 3.5])

    ax = plot_mean_ci(x, mean_y, ci_lower, ci_upper)

    assert ax is not None


def test_plot_mean_std_smoke():
    x = np.array([1, 2, 3])
    mean_y = np.array([1.0, 2.0, 3.0])
    std_y = np.array([0.1, 0.2, 0.3])

    ax = plot_mean_std(x, mean_y, std_y)

    assert ax is not None
