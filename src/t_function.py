from math import exp

import numpy as np

from plot import orange
from plot_util import PlotWrapper

r = 1

def t_negative_lower_function(c: float) -> float:
    return exp(c - 1) * c


def t_lower_function(c: float) -> float:
    return (1 - c)/t_negative_lower_function(c)


def t_upper_function(c: float) -> float:
    return (1 - c)/(t_negative_lower_function(c) + exp(-1) * c ** 2 / (2 * (1 - c)))


def t_another_function(c: float) -> float:
    return (1 - c)/(t_negative_lower_function(c) * (1 + c))


def plot_t_function(min_c: float, upper=0.5, name="min_c", r=1):
    lower = min_c
    plotter = PlotWrapper(min_c, upper)
    xs = np.linspace(lower, upper, 1000)
    ys_l = list(map(t_lower_function, xs))
    # ys_u = list(map(t_upper_function, xs))
    ys_u = list(map(t_another_function, xs))
    plotter.add_line(xs, ys_u, "lower", "g")
    plotter.add_line(xs, ys_l, "upper", orange)
    # plotter.add_line(xs, ys_u, "upper", "b")
    plotter.saveToFile("kf_" + name)


if __name__ == '__main__':
    plot_t_function(0.1, 0.3, "05_3")
    # plot_t_function(0.01, 0.5, "01_5")
    # plot_t_function(0.001, 0.00101, "0001_00101")
    # plot_t_function(0.00001, 0.000011, "00001_000011")
