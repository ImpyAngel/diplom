import os

import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Callable, List, Optional, Union

import random
import math

from matplotlib.axes import Axes

def plot_e(lower, upper: int, func: Callable[[int], float], math_p: float):
    x = np.linspace(lower, upper)
    y = list(map(func, x))
    print(y)
    plt.figure()
    plt.plot(x, y, orange)
    plt.hlines(math_p, lower, upper)
    plt.grid(True)
    plt.show()
    plt.close()


def plot_experiments():
    plot_e_specified(3, 3)
    # plot_reserch_for_r(100)


def plot_e_specified(r: int, delta: int):
    plot_e(abs(r) + 2, 1000, lambda n: pow(1. + r / n, n + delta), math.exp(r))


def plot_reserch_for_r(n: int):
    plot_e(1, round(n/2 - 1), lambda r: pow(1. - r / n, r), pow(1/2, n))


def pred_T_first(c: float) -> float:
    return 1./(math.exp(c - 1) * c)


def pred_T_second(c: float) -> float:
    return 1./(math.exp(c - 1) * c + math.exp(-1) * (1. - c) * c ** 2)


if __name__ == '__main__':
    c = 0.095
    print(pred_T_first(c))
    print(pred_T_second(c))
    # plot_T_bigger_X(1, 100, 20)
